Relation-Shape Convolutional Neural Network for Point Cloud Analysis
===
This repository contains the code (__comming soon__) in Pytorch for the paper:

__Relation-Shape Convolutional Neural Network for Point Cloud Analysis__ ([arXiv](kkkkkk))
<br>
[Yongcheng Liu](https://yochengliu.github.io/), [Bin Fan](http://www.nlpr.ia.ac.cn/fanbin/), [Shiming Xiang](https://scholar.google.com/citations?user=0ggsACEAAAAJ&hl=zh-CN) and [Chunhong Pan](http://people.ucas.ac.cn/~0005314)
<br>
[__CVPR 2019 Oral__](http://cvpr2019.thecvf.com/) &nbsp;&nbsp;&nbsp; __Project Page__: [https://yochengliu.github.io/Relation-Shape-CNN/](https://yochengliu.github.io/Relation-Shape-CNN/)


[example_results]: ./images/partseg.jpg
![example_results]
<p align = 'center'>
    <small>Example results on MS-COCO and NUS-WIDE "with" and "without" knowledge distillation using our proposed framework. The texts on the right are the top-3 predictions, where correct ones are shown in blue and incorrect in red. The green bounding boxes in images are the top-10 proposals detected by the weakly-supervised detection model.</small>
</p>

# Abstract   

Multi-label image classification (__MLIC__) is a fundamental but challenging task towards general visual understanding. Existing methods found the region-level cues (e.g., features from RoIs) can facilitate multi-label classification. Nevertheless, such methods usually require laborious object-level annotations (i.e., object labels and bounding boxes) for effective learning of the object-level visual features. In this paper, we propose a novel and efficient deep framework to boost multi-label classification by distilling knowledge from weakly-supervised detection task ___without bounding box annotations___. Specifically, given the image-level annotations, __(1)__ we first develop a weakly-supervised detection (__WSD__) model, and then __(2)__ construct an end-to-end multi-label image classification framework augmented by a knowledge distillation module that guides the classification model by the WSD model according to the class-level predictions for the whole image and the object-level visual features for object RoIs. The WSD model is the ___teacher___ model and the classification model is the ___student___ model. After this ___cross-task knowledge distillation___, the performance of the classification model is significantly improved and the efficiency is maintained since the WSD model can be safely discarded in the test phase. Extensive experiments on two large-scale datasets (MS-COCO and NUS-WIDE) show that our framework achieves superior performances over the state-of-the-art methods on both performance and efficiency.

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
