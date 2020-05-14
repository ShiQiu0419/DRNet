# Dense-Resolution Network for Point Cloud Classification and Segmentation
This repository is for Dense-Resolution Networ (DRNet) introduced in the following paper

[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/), "Dense-Resolution Network for Point Cloud Classification and Segmentation" 

## Paper
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/1911.12885)

## Introduction
Point cloud analysis is attracting attention from Artificial Intelligence research since it can be extensively applied for robotics, Augmented Reality, self-driving, etc. However, it is always challenging due to problems such as irregularities, unorderedness, and sparsity. In this article, we propose a novel network named Dense-Resolution Network for point cloud analysis. This network is designed to learn local point features from point cloud in different resolutions. In order to learn local point groups more intelligently, we present a novel grouping algorithm for local neighborhood searching and an effective error-minimizing model for capturing local features. In addition to validating the network on widely used point cloud segmentation and classification benchmarks, we also test and visualize the performances of the components. Comparing with other state-of-the-art methods, our network shows superiority.

## Motivation
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/DRNet/blob/master/intro.png">
</p>

## Implementation
* Python 3.6
* Pytorch 1.3.0
* Cuda 10.0

## Experimental Results
**Segmentation Task: [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)**
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/DRNet/blob/master/partseg.png">
</p>

**Classification Task: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/)**
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/DRNet/blob/master/cls.png">
</p>

## Visualization
**Adaptive Dilated Point Grouping (ADPG) algorithm**

*First-row: the learned dilation factors in the shallow layer of our network.* 

*Second-row: in the deep layer.*
<p align="center">
  <img width="800" src="https://github.com/ShiQiu0419/DRNet/blob/master/visual.png">
</p>

## Citation

If you find our paper is useful, please cite:

        @article{qiu2019geometric,
          title={Dense-Resolution Network for Point Cloud Classification and Segmentation},
          author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
          journal={arXiv preprint arXiv:1911.12885},
          year={2019}
        }

## Codes
**The codes will be released Soon**
