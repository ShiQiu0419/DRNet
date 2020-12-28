import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import DRNET_Seg as DRNET
from data import ShapeNetPart
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='DRNET Shape Part Segmentation Training')
parser.add_argument('--config', default='cfgs/config_partseg_gpus.yaml', type=str)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    
    train_dataset = ShapeNetPart(root = args.data_root, num_points = args.num_points, split = 'trainval', normalize = True, transforms = train_transforms)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers),
        pin_memory=True
    )
    
    global test_dataset
    test_dataset = ShapeNetPart(root = args.data_root, num_points = args.num_points, split = 'test', normalize = True, transforms = test_transforms)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers),
        pin_memory=True
    )
    
    device = torch.device("cuda")
    model = DRNET(num_classes = args.num_classes).to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    '''
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0.001)
    '''
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset)/args.batch_size
    
    # training
    train(train_dataloader, test_dataloader, model, device, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)

    
def train(train_dataloader, test_dataloader, model, device, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudAug = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global Class_mIoU, Inst_mIoU
    Class_mIoU, Inst_mIoU = 0.82, 0.85
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        # scheduler.step()
        for i, data in enumerate(train_dataloader, 0):
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            
            points, target, cls = data
            points, target = Variable(points), Variable(target)
            points, target = points.to(device), target.to(device)
            # augmentation
            points.data = PointcloudAug(points.data)
            
            optimizer.zero_grad()
            
            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred, error1, error2, error3, error4 = model(points, batch_one_hot_cls)
            loss1 = torch.norm(error1, dim=-1) # B, N
            loss1 = torch.mean(torch.mean(loss1, dim=-1), dim=-1)
            loss2 = torch.norm(error2, dim=-1) # B, N
            loss2 = torch.mean(torch.mean(loss2, dim=-1), dim=-1)
            loss3 = torch.norm(error3, dim=-1) # B, N
            loss3 = torch.mean(torch.mean(loss3, dim=-1), dim=-1)
            loss4 = torch.norm(error4, dim=-1) # B, N
            loss4 = torch.mean(torch.mean(loss4, dim=-1), dim=-1)
            pred = pred.view(-1, args.num_classes)
            target = target.view(-1,1)[:,0]
            loss = criterion(pred, target) + 0.1*loss1
            # + 0.01*loss2 + 0.01*loss3 + 0.01*loss4
            loss.backward()
            optimizer.step()
            
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if (epoch >60) and args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                print('testing..')
                validate(test_dataloader, model, device, criterion, args, batch_count)

def validate(test_dataloader, model, device, criterion, args, iter): 
    global Class_mIoU, Inst_mIoU, test_dataset
    model.eval()
    
    seg_classes = test_dataset.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {}           # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    losses = []
    temp_device = torch.device("cpu")
    for _, data in enumerate(test_dataloader, 0):
        points, target, cls = data
        with torch.no_grad():
            points = Variable(points)
        with torch.no_grad():
            target = Variable(target)

        points = points.to(device) 
        target = target.to(device)

        batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())
        
        with torch.no_grad():
            pred, error1, error2, error3, error4 = model(points, batch_one_hot_cls)

        loss1 = torch.norm(error1, dim=-1) # B, N
        loss1 = torch.mean(torch.mean(loss1, dim=-1), dim=-1)
        loss2 = torch.norm(error2, dim=-1) # B, N
        loss2 = torch.mean(torch.mean(loss2, dim=-1), dim=-1)
        loss3 = torch.norm(error3, dim=-1) # B, N
        loss3 = torch.mean(torch.mean(loss3, dim=-1), dim=-1)
        loss4 = torch.norm(error4, dim=-1) # B, N
        loss4 = torch.mean(torch.mean(loss4, dim=-1), dim=-1)
        loss = criterion(pred.view(-1, args.num_classes), target.view(-1,1)[:,0]) + 0.1*loss1
        # + 0.01*loss2 + 0.01*loss3 + 0.01*loss4
        losses.append(loss.data.clone())

        pred = pred.data.cpu()
        target = target.data.cpu()

        pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(len(cls)):
            cat = seg_label_to_cat[target[b, 0].item()]
            logits = pred[b, :, :]   # (num_points, num_classes)
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]
        
        for b in range(len(cls)):
            segp = pred_val[b, :].to(temp_device)
            segl = target[b, :].to(temp_device)
            cat = seg_label_to_cat[segl[0].item()]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if torch.sum((segl == l) | (segp == l)) == 0:
                    # part is not present in this shape
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = torch.sum((segl == l) & (segp == l)) / float(torch.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))  # torch.mean(torch.stack(part_ious))
        
    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_class_ious = np.mean(list(shape_ious.values()))
    
    for cat in sorted(shape_ious.keys()):
        print('****** %s: %0.6f'%(cat, shape_ious[cat]))
    print('************ Test Loss: %0.6f' % (torch.mean(torch.stack(losses)).cpu().numpy())) #torch.mean(torch.stack(losses)).numpy() np.array(losses).mean())
    print('************ Class_mIoU: %0.6f' % (mean_class_ious))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))

    if mean_class_ious > Class_mIoU or np.mean(instance_ious) > Inst_mIoU:
        if mean_class_ious > Class_mIoU:
            Class_mIoU = mean_class_ious
        if np.mean(instance_ious) > Inst_mIoU:
            Inst_mIoU = np.mean(instance_ious)
        torch.save(model.state_dict(), '%s/seg_drnet_iter_%d_ins_%0.6f_cls_%0.6f.pth' % (args.save_path, iter, np.mean(instance_ious), mean_class_ious))
    model.train()
    
if __name__ == "__main__":
    main()
