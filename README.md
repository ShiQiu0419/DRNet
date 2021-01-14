# Dense-Resolution Network for Point Cloud Classification and Segmentation
This repository is for Dense-Resolution Networ (DRNet) introduced in the following paper

[Shi Qiu](https://shiqiu0419.github.io/) [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)  
"Dense-Resolution Network for Point Cloud Classification and Segmentation"  
acceptd in IEEE Winter Conference on Applications of Computer Vision (WACV 2021)

## Paper
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2005.06734) or [here (CVF)](https://openaccess.thecvf.com/content/WACV2021/papers/Qiu_Dense-Resolution_Network_for_Point_Cloud_Classification_and_Segmentation_WACV_2021_paper.pdf), together with [supplementary material](https://openaccess.thecvf.com/content/WACV2021/supplemental/Qiu_Dense-Resolution_Network_for_WACV_2021_supplemental.pdf).

## Motivation
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/DRNet/blob/master/figures/intro.png">
</p>

## Implementation
* Python 3.6
* Pytorch 1.3.0
* Cuda 10.0

## Dataset
Download the [ShapeNet Part Dataset](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and upzip it to your rootpath. Alternatively, you can modify the path of your dataset in `cfgs/config_partseg_gpus.yaml` and `cfgs/config_partseg_test.yaml`.

## CUDA Kernel Building
For PyTorch version <= 0.4.0, please refer to [Relation-Shape-CNN](https://github.com/Yochengliu/Relation-Shape-CNN).  
For PyTorch version >= 1.0.0, please refer to [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).  

**Note:**  
In our DRNet, we use Farthest Point Sampling (e.g., `pointnet2_utils.furthest_point_sample`) to down-sample the point cloud. Also, we adpot Feature Propagation (e.g., `pointnet2_utils.three_nn` and `pointnet2_utils.three_interpolate`) to up-sample the feature maps.

## Training

    sh train_partseg_gpus.sh
        
Due to the complexity of DRNet, we support Multi-GPU via `nn.DataParallel`. You can also adjust other parameters such as batch size or the number of input points in `cfgs/config_partseg_gpus.yaml`, in order to fit the memory limit of your device.

## Voting Evaluation
You can set the path of your pre-trained model in `cfgs/config_partseg_test.yaml`, then run:

    sh voting_test.sh
   
## Citation

If you find our paper is useful, please cite:

    @inproceedings{qiu2021dense,
      title={Dense-Resolution Network for Point Cloud Classification and Segmentation},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      month={January},
      year={2021},
      pages={3813-3822}
    }

## Acknowledgement
The code is built on [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [Relation-Shape-CNN](https://github.com/Yochengliu/Relation-Shape-CNN), [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch). We thank the authors for sharing their codes.
