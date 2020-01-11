Relation-Shape Convolutional Neural Network for Point Cloud Analysis
===
This repository contains the author's implementation in Pytorch for the paper:

__Relation-Shape Convolutional Neural Network for Point Cloud Analysis__ [[arXiv](https://arxiv.org/abs/1904.07601)] [[CVF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.pdf)]
<br>
[Yongcheng Liu](https://yochengliu.github.io/), [Bin Fan](http://www.nlpr.ia.ac.cn/fanbin/), [Shiming Xiang](https://scholar.google.com/citations?user=0ggsACEAAAAJ&hl=zh-CN) and [Chunhong Pan](http://people.ucas.ac.cn/~0005314)
<br>
[__CVPR 2019 Oral & Best paper finalist__](http://cvpr2019.thecvf.com/) &nbsp;&nbsp;&nbsp; __Project Page__: [https://yochengliu.github.io/Relation-Shape-CNN/](https://yochengliu.github.io/Relation-Shape-CNN/)

## Citation

If our paper is helpful for your research, please consider citing:   
```BibTex
        @inproceedings{liu2019rscnn,   
            author = {Yongcheng Liu and    
                            Bin Fan and    
                      Shiming Xiang and   
                           Chunhong Pan},   
            title = {Relation-Shape Convolutional Neural Network for Point Cloud Analysis},   
            booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},    
            pages = {8895--8904},  
            year = {2019}   
        }   
```
## Usage: Preparation

### Requirement

- Ubuntu 14.04
- Python 3 (recommend Anaconda3)
- Pytorch 0.3.\*/0.4.\*
- CMake > 2.8
- CUDA 8.0 + cuDNN 5.1

### Building Kernel

    git clone https://github.com/Yochengliu/Relation-Shape-CNN.git 
    cd Relation-Shape-CNN

- mkdir build && cd build
- cmake .. && make

### Dataset
__Shape Classification__

Download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) (415M). Replace `$data_root$` in `cfgs/config_*_cls.yaml` with the dataset parent path.

__ShapeNet Part Segmentation__

Download and unzip [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) (674M). Replace `$data_root$` in `cfgs/config_*_partseg.yaml` with the dataset path.

## Usage: Training
### Shape Classification

    sh train_cls.sh
        
You can modify `relation_prior` in `cfgs/config_*_cls.yaml`. We have trained a Single-Scale-Neighborhood classification model in `cls` folder, whose accuracy is 92.38%.
        
### Shape Part Segmentation

    sh train_partseg.sh
        
We have trained a Multi-Scale-Neighborhood part segmentation model in `seg` folder, whose class mIoU and instance mIoU is 84.18% and 85.81% respectively.

## Usage: Evaluation
### Shape Classification

    Voting script: voting_evaluate_cls.py
        
You can use our model `cls/model_cls_ssn_iter_16218_acc_0.923825.pth` as the checkpoint in `config_ssn_cls.yaml`, and after this voting you will get an accuracy of 92.71% if all things go right.

### Shape Part Segmentation

    Voting script: voting_evaluate_partseg.py
        
You can use our model `seg/model_seg_msn_iter_57585_ins_0.858054_cls_0.841787.pth` as the checkpoint in `config_msn_partseg.yaml`.

## License

The code is released under MIT License (see LICENSE file for details).

## Acknowledgement

The code is heavily borrowed from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
        
## Contact

If you have some ideas or questions about our research to share with us, please contact <yongcheng.liu@nlpr.ia.ac.cn>
