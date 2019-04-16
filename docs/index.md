<h1 align = "center">RS-CNN: Relation-Shape Convolutional Neural Network for Point Cloud Analysis</h1>
<p align = "center">
    <a href="https://yochengliu.github.io/">Yongcheng Liu</a>
</p>
<br>
<p align = "center">
    <font size=5>[__CVPR 2019__](http://cvpr2019.thecvf.com/)</font><font size=5 color=#FF4500> __Oral Presentation__</font>
</p>
<br>

[partseg_results]: ./images/partseg.jpg
![partseg_results]
<p align = 'center'>
    <font size=3>Segmentation examples on ShapeNet part benchmark. Although the part shapes implied in irregular points are extremely diverse and they may be very confusing to recognize, our RS-CNN can also segment them out with decent accuracy.</font>
</p>

# Abstract   

Point cloud analysis is very challenging, as the shape implied in irregular points is difficult to capture. In this paper, we propose RS-CNN, namely, Relation-Shape Convolutional Neural Network, which extends regular grid CNN to irregular configuration for point cloud analysis. The key to RS-CNN is learning from relation, i.e., the geometric topology constraint among points. Specifically, the convolutional weight for local point set is forced to learn a high-level relation expression from predefined geometric priors, between a sampled point from this point set and the others. In this way, an inductive local representation with explicit reasoning about the spatial layout of points can be obtained, which leads to much shape awareness and robustness. With this convolution as a basic operator, RS-CNN, a hierarchical architecture can be developed to achieve contextual shape-aware learning for point cloud analysis. Extensive experiments on challenging benchmarks across three tasks verify RS-CNN achieves the state of the arts.

# Motivation

[motivation]: ./images/motivation.jpg
![motivation]
<p align = 'center'>
<small>Correct predictions are shown in blue and incorrect in red.</small>
</p>

- The MLIC model might not predict well due to the lack of ___object-level feature extraction___ and ___localization for multiple semantic instances___.

- Although the results detected by WSD may not preserve object boundaries well, they tend to ___locate the semantic regions___ which are ___informative for classifying the target object___, such that the predictions can still be improved.

- Therefore, the localization results of WSD could provide ___object-relevant semantic regions___ while its image-level predictions could naturally capture ___the latent class dependencies___. These unique advantages are very useful for the MLIC task.

# Framework

[framework]: ./images/framework.png
![framework]
<p align = 'center'>
<small>The proposed framework works with two steps: (1) we first develop a WSD model as teacher model (called T-WDet) with only image-level annotations y; (2) then the knowledge in T-WDet is distilled into the MLIC student model (called S-Cls) via feature-level distillation from RoIs and prediction-level distillation from the whole image, where the former is conducted by optimizing the loss in Eq. (3) while the latter is conducted by optimizing the losses in Eq. (5) and Eq. (10). </small>
</p>

In this paper, we propose a novel and efficient deep framework to boost MLIC by ___distilling the unique knowledge from WSD into classification with only image-level annotations___.

Specifically, our framework works with ___two steps___:

- __(1)__ we first develop a WSD model with image-level annotations; 
- __(2)__ then we construct an ___end-to-end knowledge distillation framework___ by propagating the ___class-level holistic predictions___ and ___the object-level features from RoIs___ in the WSD model to the MLIC model, where the WSD model is taken as the teacher model (called __T-WDet__) and the classification model is the student model (called __S-Cls__).


- The distillation of object-level features from RoIs focuses on ___perceiving localizations of semantic regions___ detected by the WSD model while the distillation of class-level holistic predictions aims at ___capturing class dependencies___ predicted by the WSD model.

- After this distillation, the classification model could be significantly improved and ___no longer need the WSD model___, thus resulting in ___high efficiency___ in test phase. More details can be referred in the paper.

# Ablation Study

### Overall Ablation

[overall_ab]: ./images/overall_ab.jpg
![overall_ab]

- The T-WDet model achieves very good performance on MS-COCO while slightly better performance on NUS-WIDE. The reason may be that the clean object labels on MS-COCO are quite suitable for detection task while the noisy concept labels are not.

- After distillation, the MLIC model not only has global information learned by itself, but also perceives the local semantic regions as ___complementary cues___ distilled from the WSD model, thus it could surpass the latter on NUS-WIDE.

### Region Proposal

[proposal]: ./images/proposal.jpg
![proposal]

- The classification performance of T-WDet is improved from 78.6 to 81.1 when using the fully-supervised detection results (__Faster-RCNN__).

- The S-Cls model is improved to 76.3 compared with EdgeBoxes proposals to 74.6, where the gap is not obvious. This further demonstrates the effectiveness and practicability of our proposed framework.

### Robustness 

[coco]: ./images/coco.png
![coco]
[nus]: ./images/nus.png
![nus]
<p align = 'center'><small>The improvements of S-Cls model over each class/concept on MS-COCO (upper figure) and NUS-WIDE (lower figure) after knowledge distillation with our framework. "*k" indicates the number (divided by 1000) of images including this class/concept. The classes/concepts in horizontal axis are sorted by the number "*k" from large to small.</small></p>

- The improvements are also considerable even when the classes are very __imbalanced__ (on NUS-WIDE, the classes in which the number of images is fewer are improved even more).

- The improvements are robust to the ___object's size___ and the ___label's type___. On MS-COCO, small objects like "bottle", "fork", "apple" and so on, which may be difficult for the classification model to pay attention, are also improved a lot. On NUS-WIDE, scenes (e.g., "rainbow"), events (e.g., "earthquake") and objects (e.g., "book") are all improved considerably.

# Code

Please refer to the [GitHub repository](https://github.com/Yochengliu/MLIC-KD-WSD) for more details. 

# Publication

Yongcheng Liu, Lu Sheng, Jing Shao, Junjie Yan, Shiming Xiang and Chunhong Pan, "Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection", in ACM International Conference on Multimedia (MM), 2018. [[ACM DL](https://dl.acm.org/citation.cfm?id=3240567)] [[arXiv](https://arxiv.org/abs/1809.05884)]

```
@inproceedings{liu2019rscnn,   
  author = {Yongcheng Liu and    
            Bin Fan and    
            Shiming Xiang and   
            Chunhong Pan},   
  title = {Relation-Shape Convolutional Neural Network for Point Cloud Analysis},   
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},    
  pages = {1--10},  
  year = {2019}   
}   
```
