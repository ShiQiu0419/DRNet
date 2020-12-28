# Dense-Resolution Network for Point Cloud Classification and Segmentation
This repository is for Dense-Resolution Networ (DRNet) introduced in the following paper

[Shi Qiu](https://shiqiu0419.github.io/) [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)  
"Dense-Resolution Network for Point Cloud Classification and Segmentation"  
Acceptd in *IEEE Winter Conference on Applications of Computer Vision* (WACV 2021)

## Paper
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2005.06734)

## Introduction
Point cloud analysis is attracting attention from Artificial Intelligence research since it can be widely used in applications such as robotics, Augmented Reality, self-driving. However, it is always challenging due to irregularities, unorderedness, and sparsity. In this article, we propose a novel network named Dense-Resolution Network (DRNet) for point cloud analysis. Our DRNet is designed to learn local point features from the point cloud in different resolutions. In order to learn local point groups more effectively, we present a novel grouping method for local neighborhood searching and an error-minimizing module for capturing local features. In addition to validating the network on widely used point cloud segmentation and classification benchmarks, we also test and visualize the performance of the components. Comparing with other state-of-the-art methods, our network shows superiority on ModelNet40, ShapeNet synthetic and ScanObjectNN real point cloud datasets.

## Motivation
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/intro.png">
</p>

## Implementation
* Python 3.6
* Pytorch 1.3.0
* Cuda 10.0

## Experimental Results
**Segmentation Task: [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)**
<p align="center">
  <img width="900" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/shapenet.png">
</p>

**Classification Task: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/)**
<p align="center">
  <img width="450" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/modelnet.png">
</p>
<p align="center">
  <img width="900" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/scanobjectnn.png">
</p>

## Visualization
**Adaptive Dilated Point Grouping (ADPG) algorithm**

*First-row: the learned dilation factors in the shallow layer of our network.* 

*Second-row: in the deep layer.*
<p align="center">
  <img width="800" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/visual.png">
</p>

**Comparisons**
<p align="center">
  <img width="800" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/compare.png">
</p>

## Dataset
Download the [ShapeNet Part Dataset](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and upzip it to your rootpath. Alternatively, you can modify the path of your dataset in `cfgs/config_partseg_gpus.yaml` and `cfgs/config_partseg_test.yaml`.

## CUDA Kernel Building
For PyTorch version <= 0.4.0, please refer to [Relation-Shape-CNN](https://github.com/Yochengliu/Relation-Shape-CNN).  
For PyTorch version >= 1.0.0, please refer to [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).  

**Note**  
In our DRNet, we require Farthest Point Sampling (`pointnet2_utils.furthest_point_sample`) to down-sample the point cloud. Also, we adpot Feature Propagation (`pointnet2_utils.three_nn` and `pointnet2_utils.three_interpolate`) to up-sample the feature maps.

## Training

    sh train_partseg_gpus.sh
        
Due to the complexity of DRNet, we support Multi-GPU via `nn.DataParallel`. You can also adjust other parameters such as batch size or the number of input points in `cfgs/config_partseg_gpus.yaml`, in oder to fit the memory limit of your device.

## Citation

If you find our paper is useful, please cite:

    @inproceedings{qiu2021dense,
      title={Dense-Resolution Network for Point Cloud Classification and Segmentation},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      booktitle={The IEEE Winter Conference on Applications of Computer Vision},
      year={2021}
    }

## Acknowledgement
We thank authors of [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [Relation-Shape-CNN](https://github.com/Yochengliu/Relation-Shape-CNN), [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) for sharing their codes.
