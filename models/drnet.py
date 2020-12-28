#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.utils import pointnet2_utils


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def pw_dist(x):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (batch_size, num_points, n)

    return -pairwise_distance


def knn_metric(x, d, conv_op1, conv_op2, conv_op11, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    metric = (-pairwise_distance).topk(k=d*k, dim=-1, largest=False)[0] # B,N,100
    metric_idx = (-pairwise_distance).topk(k=d*k, dim=-1, largest=False)[1] # B,N,100
    metric_trans = metric.permute(0, 2, 1) # B,100,N
    metric = conv_op1(metric_trans) # B,50,N
    metric = torch.squeeze(conv_op11(metric).permute(0, 2, 1), -1) # B,N
    # normalize function
    metric = torch.sigmoid(-metric)
    # projection function
    metric = 5 * metric + 0.5
    # scaling function
    
    value1 = torch.where((metric>=0.5)&(metric<1.5), torch.full_like(metric, 1), torch.full_like(metric, 0))
    value2 = torch.where((metric>=1.5)&(metric<2.5), torch.full_like(metric, 2), torch.full_like(metric, 0))
    value3 = torch.where((metric>=2.5)&(metric<3.5), torch.full_like(metric, 3), torch.full_like(metric, 0))
    value4 = torch.where((metric>=3.5)&(metric<4.5), torch.full_like(metric, 4), torch.full_like(metric, 0))
    value5 = torch.where((metric>=4.5)&(metric<=5.5), torch.full_like(metric, 5), torch.full_like(metric, 0))

    value = value1 + value2 + value3 + value4 + value5 # B,N
    
    select_idx = torch.cuda.LongTensor(np.arange(k)) # k
    select_idx = torch.unsqueeze(select_idx, 0).repeat(num_points, 1) # N,k
    select_idx = torch.unsqueeze(select_idx, 0).repeat(batch_size, 1, 1) # B,N,k
    value = torch.unsqueeze(value, -1).repeat(1, 1, k) # B,N,k
    select_idx = select_idx * value
    select_idx = select_idx.long()
    idx = pairwise_distance.topk(k=k*d, dim=-1)[1]   # (batch_size, num_points, k*d)
    # dilatedly selecting k from k*d idx
    idx = torch.gather(idx, dim=-1, index=select_idx) # B,N,k
    return idx


def get_adptive_dilated_graph_feature(x, conv_op1, conv_op2, conv_op11, d=5, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_metric(x, d, conv_op1, conv_op2, conv_op11, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device)
    idx_base = idx_base.view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device)
    idx_base = idx_base.view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class Channel_fusion(nn.Module):
    def __init__(self, in_dim):
        super(Channel_fusion, self).__init__()
        self.channel_in = in_dim

        self.bn1 = nn.BatchNorm1d(in_dim//8)
        self.bn2 = nn.BatchNorm1d(in_dim)

        self.squeeze_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, bias=False),
                                    self.bn1)
        self.excite_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim//8, out_channels=in_dim, kernel_size=1, bias=False),
                                    self.bn2)

    def forward(self, x):
        """
            inputs :
                x : B, C, N
            returns :
                out : B, C, N
        """
        batch_size, channel_num, num_points= x.size()
        fusion_score = F.adaptive_avg_pool1d(x, 1) # B, C, 1
        fusion_score = self.squeeze_conv(fusion_score) # B, C', 1
        fusion_score = F.relu(fusion_score)
        fusion_score = self.excite_conv(fusion_score) # B, C, 1
        fusion_score = torch.sigmoid(fusion_score).expand_as(x) # B, C, N

        return fusion_score


class DRNET(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(3)
        self.bn12 = nn.BatchNorm1d(512)
        self.bn13 = nn.BatchNorm1d(1024)
        self.bn14 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn21 = nn.BatchNorm1d(256)
        self.bn22 = nn.BatchNorm1d(128)
        self.bn23 = nn.BatchNorm1d(128)
        self.bn24 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm1d(256)
        self.bn33 = nn.BatchNorm1d(256)
        self.bn34 = nn.BatchNorm1d(128)
        self.bn35 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn53 = nn.BatchNorm1d(512)
        self.bn54 = nn.BatchNorm1d(1024)
        self.bn63 = nn.BatchNorm1d(128)
        self.bn64 = nn.BatchNorm1d(50)
        self.bn7 = nn.BatchNorm1d(1024)
        
        self.bn8 = nn.BatchNorm1d(1024)
        self.bn9 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=[1,20], bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv13 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn13,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv14 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn14,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv21 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn21,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv22 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn22,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv23 = nn.Sequential(nn.Conv1d(192, 128, kernel_size=1, bias=False),
                                   self.bn23,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv24 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn24,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv32 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn32,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv33 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn33,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv34 = nn.Sequential(nn.Conv1d(384, 128, kernel_size=1, bias=False),
                                   self.bn34,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv35 = nn.Sequential(nn.Conv1d(320, 128, kernel_size=1, bias=False),
                                   self.bn35,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv53 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                   self.bn53,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv54 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn54,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv63 = nn.Sequential(nn.Conv1d(1024, 128, kernel_size=1, bias=False),
                                   self.bn63,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv64 = nn.Sequential(nn.Conv1d(128, 50, kernel_size=1))

        self.conv7 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(2112, 1024, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))       

        self.bnfc2 = nn.BatchNorm2d(64)
        self.bnfc21 = nn.BatchNorm2d(64)
        self.bnfc24 = nn.BatchNorm2d(64)
        self.bnfc3 = nn.BatchNorm2d(128)
        self.bnfc31 = nn.BatchNorm2d(64)
        self.bnfc4 = nn.BatchNorm2d(256)
        self.bnfc41 = nn.BatchNorm2d(128)
        self.convfc2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bnfc2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=[1,20], bias=False),
                                   self.bnfc21,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc24 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bnfc24,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bnfc3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc31 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=[1,20], bias=False),
                                   self.bnfc31,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bnfc4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convfc41 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=[1,20], bias=False),
                                   self.bnfc41,
                                   nn.LeakyReLU(negative_slope=0.2))    

        self.bnop1 = nn.BatchNorm1d(50)
        self.bnop2 = nn.BatchNorm1d(50)
        self.bnop3 = nn.BatchNorm1d(50)
        self.bnop4 = nn.BatchNorm1d(50)
        self.bnop11 = nn.BatchNorm1d(1)
        self.bnop21 = nn.BatchNorm1d(1)
        self.bnop31 = nn.BatchNorm1d(1)
        self.bnop41 = nn.BatchNorm1d(1)
        self.bnop12 = nn.BatchNorm1d(1)
        self.bnop22 = nn.BatchNorm1d(1)
        self.bnop32 = nn.BatchNorm1d(1)
        self.bnop42 = nn.BatchNorm1d(1)
        self.conv_op1 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1), self.bnop1, nn.LeakyReLU(negative_slope=0.2))
        self.conv_op2 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1), self.bnop2, nn.LeakyReLU(negative_slope=0.2))
        self.conv_op3 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1), self.bnop3, nn.LeakyReLU(negative_slope=0.2))
        self.conv_op4 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1), self.bnop4, nn.LeakyReLU(negative_slope=0.2))
        self.conv_op11 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=1, kernel_size=1), self.bnop11)
        self.conv_op21 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=1, kernel_size=1), self.bnop21)
        self.conv_op31 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=1, kernel_size=1), self.bnop31)
        self.conv_op41 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=1, kernel_size=1), self.bnop41)
        self.conv_op12 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=1, kernel_size=1), self.bnop12)
        self.conv_op22 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=1, kernel_size=1), self.bnop22)
        self.conv_op32 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=1, kernel_size=1), self.bnop32)
        self.conv_op42 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=1, kernel_size=1), self.bnop42)

        self.fuse = Channel_fusion(1024)

        self.dp = nn.Dropout(p=0.5)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, cls):
        # x: B,3,N

        xyz, features = self._break_up_pc(pointcloud)
        num_pts = xyz.size(1)
        batch_size = xyz.size(0)
        # FPS to find different point subsets and their relations 
        subset1_idx = pointnet2_utils.furthest_point_sample(xyz, num_pts//4).long() # B,N/2
        subset1_xyz = torch.unsqueeze(subset1_idx, -1).repeat(1, 1, 3) # B,N/2,3
        subset1_xyz = torch.take(xyz, subset1_xyz) # B,N/2,3

        dist, idx1 = pointnet2_utils.three_nn(xyz, subset1_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight1 = dist_recip / norm

        subset12_idx = pointnet2_utils.furthest_point_sample(subset1_xyz, num_pts//16).long() # B,N/4
        subset12_xyz = torch.unsqueeze(subset12_idx, -1).repeat(1, 1, 3) # B,N/4,3
        subset12_xyz = torch.take(subset1_xyz, subset12_xyz) # B,N/4,3

        dist, idx12 = pointnet2_utils.three_nn(subset1_xyz, subset12_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight12 = dist_recip / norm

        device = torch.device('cuda')
        centroid = torch.zeros([batch_size, 1, 3], device=device)
        dist, idx0 = pointnet2_utils.three_nn(subset12_xyz, centroid)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight0 = dist_recip / norm
        #######################################
        # Error-minimizing module 1:
        # Encoding
        x = xyz.transpose(2, 1) # x: B,3,N
        x1_1 = x
        x = get_adptive_dilated_graph_feature(x, self.conv_op1, self.conv_op11, self.conv_op12, d=5, k=20)
        x = self.conv1(x)  # B,64,N,k
        x = self.conv14(x) # B,64,N,k
        x1_2 = x
        # Back-projection
        x = self.conv11(x) # B,3,N,1
        x = torch.squeeze(x, -1) # B,3,N
        x1_3 = x
        # Calculating Error
        delta_1 = x1_3 - x1_1 # B,3,N
        # Output
        x = x1_2 # B,64,N,k
        x1 = x.max(dim=-1, keepdim=False)[0]  # B,64,N
        #######################################

        #######################################
        # Multi-resolution (MR) Branch
        # Down-scaling 1
        subset1_feat = torch.unsqueeze(subset1_idx, -1).repeat(1, 1, 64) # B,N/2,64
        x1_subset1 = torch.take(x1.transpose(1, 2).contiguous(), subset1_feat).transpose(1, 2).contiguous() # B,64,N/2

        x2_1 = x1_subset1 # B,64,N/2
        x = get_graph_feature(x1_subset1, k=self.k//2)
        x = self.conv2(x)  # B,64,N/2,k
        x = self.conv24(x)  # B,128,N/2,k
        x2 = x.max(dim=-1, keepdim=False)[0]  # B,128,N/2

        # Dense-connection
        x12 = pointnet2_utils.three_interpolate(x2, idx1, weight1) # B,128,N
        x12 = torch.cat((x12, x1), dim=1) # B,192,N
        x12 = self.conv23(x12) # B,128,N

        # Down-scaling 2
        subset12_feat = torch.unsqueeze(subset12_idx, -1).repeat(1, 1, 128) # B,N/4,128
        x2_subset12 = torch.take(x2.transpose(1, 2).contiguous(), subset12_feat).transpose(1, 2).contiguous() # B,128,N/4

        x3_1 = x2_subset12 # B,128,N/4
        x = get_graph_feature(x2_subset12, k=self.k//4)
        x = self.conv3(x)  # B,256,N/4,k
        x3 = x.max(dim=-1, keepdim=False)[0]  # B,256,N/4

        # Dense-connection
        x23 = pointnet2_utils.three_interpolate(x3, idx12, weight12) # B,256,N/2
        x23 = torch.cat((x23, x2), dim=1) # B,384,N/2
        x23 = self.conv34(x23) # B,128,N/2
        x123 = pointnet2_utils.three_interpolate(x23, idx1, weight1) # B,128,N
        x123 = torch.cat((x123, x12, x1), dim=1) # B,320,N
        x123 = self.conv35(x123) # B,128,N       

        # Down-scaling 3
        x_bot = self.conv53(x3)
        x_bot = self.conv54(x_bot) # B,1024,N/128
        x_bot = F.adaptive_max_pool1d(x_bot, 1)  # B,1024,1

        # Upsampling 3:
        interpolated_feats1 = pointnet2_utils.three_interpolate(x_bot, idx0, weight0) # B,1024,N/4
        interpolated_feats2 = x3 # B,256,N/4
        x3_up = torch.cat((interpolated_feats1, interpolated_feats2), dim=1) # B,1280,N/4
        x3_up = self.conv32(x3_up) # B,256,N/4
        x3_up = self.conv33(x3_up) # B,256,N/4

        # Upsampling 2:
        interpolated_feats1 = pointnet2_utils.three_interpolate(x3_up, idx12, weight12) # B,256,N/2
        interpolated_feats2 = x2 # B,128,N/2
        interpolated_feats3 = x23 # B,128,N/2
        x2_up = torch.cat((interpolated_feats1, interpolated_feats3, interpolated_feats2), dim=1) # B,512,N/2
        x2_up = self.conv21(x2_up) # B,256,N/2
        x2_up = self.conv22(x2_up) # B,128,N/2

        # Upsampling 1:
        interpolated_feats1 = pointnet2_utils.three_interpolate(x2_up, idx1, weight1) # B,128,N
        interpolated_feats2 = x1 # B,64,N
        interpolated_feats3 = x12 # B,128,N
        interpolated_feats4 = x123 # B,128,N
        x1_up = torch.cat((interpolated_feats1, interpolated_feats4, interpolated_feats3, interpolated_feats2), dim=1) # B,448,N
        x1_up = self.conv12(x1_up) # B,512,N
        x1_up = self.conv13(x1_up) # B,1024,N

        x_mr = x1_up
        #############################################################################

        #############################################################################
        # Full-resolution Branch
        # Error-minimizing module 2:
        # Encoding
        x2_1 = x1 # B,64,N
        x = get_adptive_dilated_graph_feature(x1, self.conv_op2, self.conv_op21, self.conv_op22, d=5, k=20)
        x = self.convfc2(x)  # B,64,N,k
        x = self.convfc24(x)  # B,64,N,k
        x2_2 = x
        # Back-projection
        x = self.convfc21(x) # B,64,N,1
        x = torch.squeeze(x, -1) # B,64,N
        x2_3 = x
        # Calculating Error
        delta_2 = x2_3 - x2_1 # B,64,N
        # Output
        x = x2_2 # B,64,N,k
        x2 = x.max(dim=-1, keepdim=False)[0]  # B,64,N
        #######################################
        # Error-minimizing module 3:
        # Encoding
        x3_1 = x2 # B,64,N
        x = get_adptive_dilated_graph_feature(x2, self.conv_op3, self.conv_op31, self.conv_op32, d=5, k=20)
        x = self.convfc3(x)  # B,128,N,k
        x3_2 = x
        # Back-projection
        x = self.convfc31(x) # B,64,N,1
        x = torch.squeeze(x, -1) # B,64,N
        x3_3 = x
        # Calculating Error
        delta_3 = x3_3 - x3_1 # B,64,N
        # Output
        x = x3_2 # B,128,N,k
        x3 = x.max(dim=-1, keepdim=False)[0]  # B,128,N
        #######################################
        # Error-minimizing module 4:
        # Encoding
        x4_1 = x3 # B,128,N
        x = get_adptive_dilated_graph_feature(x3, self.conv_op4, self.conv_op41, self.conv_op42, d=5, k=20)
        x = self.convfc4(x)  # B,256,N,k
        x4_2 = x
        # Back-projection
        x = self.convfc41(x) # B,128,N,1
        x = torch.squeeze(x, -1) # B,128,N
        x4_3 = x
        # Calculating Error
        delta_4 = x4_3 - x4_1 # B,128,N
        # Output
        x = x4_2 # B,256,N,k
        x4 = x.max(dim=-1, keepdim=False)[0]  # B,256,N

        x = torch.cat((x1, x2, x3, x4), dim=1)  # B,512,N       
        x_fr = self.conv7(x)  # B,1024,N

        # Fusing FR and MR outputs
        fusion_score = self.fuse(x_mr)
        x = x_fr + x_fr * fusion_score
        x_all = self.conv9(x) # B,1024,N

        # Collecting global feature
        one_hot_label = cls.view(-1, 16, 1) # B,16,1
        one_hot_label = self.conv5(one_hot_label) # B,64,1
        x_max = F.adaptive_max_pool1d(x_all, 1) # B,1024,1       
        x_global = torch.cat((x_max, one_hot_label), dim=1) # B,1088,1

        x_global = x_global.repeat(1, 1, num_pts) # B,1088,N
        x = torch.cat((x_all, x_global), dim=1)  # B,2112,N

        x = self.conv8(x) # B,1024,N

        x = self.conv63(x) # B,128,N
        x = self.dp(x)
        x = self.conv64(x) # B,50,N

        return (x.transpose(2, 1).contiguous(),
        delta_1.transpose(2, 1).contiguous(),
        delta_2.transpose(2, 1).contiguous(),
        delta_3.transpose(2, 1).contiguous(),
        delta_4.transpose(2, 1).contiguous())
