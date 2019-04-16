<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>   

<h1 align = "center">Relation-Shape Convolutional Neural Network for Point Cloud Analysis</h1>
<p align = "center">
    <a href="https://yochengliu.github.io/" style="font-size: 23px">Yongcheng Liu</a> &emsp;&emsp;
    <a href="http://www.nlpr.ia.ac.cn/fanbin/" style="font-size: 23px">Bin Fan</a> &emsp;&emsp;
    <a href="https://scholar.google.com/citations?user=0ggsACEAAAAJ&hl=zh-CN" style="font-size: 23px">Shiming Xiang</a>  &emsp;&emsp;
    <a href="http://people.ucas.ac.cn/~0005314" style="font-size: 23px">Chunhong Pan</a>
</p>
<p align = "center">
    <a href="http://cvpr2019.thecvf.com/" style="font-size: 23px"><strong>CVPR 2019</strong></a> &emsp;
    <font color="red" size="5"><strong>Oral Presentation</strong></font>
</p>
<br>

[partseg_results]: ./images/partseg.jpg
![partseg_results]
<p align = 'center'>
    <small>Segmentation examples on ShapeNet part benchmark. Although the part shapes implied in irregular points are extremely diverse and they may be very confusing to recognize, our RS-CNN can also segment them out with decent accuracy.</small>
</p>

# Abstract   

Point cloud analysis is very challenging, as the shape implied in irregular points is difficult to capture. In this paper, we propose RS-CNN, namely, Relation-Shape Convolutional Neural Network, which extends regular grid CNN to irregular configuration for point cloud analysis. ___The key to RS-CNN is learning from relation___, i.e., the geometric topology constraint among points. Specifically, the convolutional weight for local point set is forced to ___learn a high-level relation expression from predefined geometric priors___, between a sampled point from this point set and the others. In this way, an inductive local representation with ___explicit reasoning about the spatial layout of points___ can be obtained, which leads to much shape awareness and robustness. With this convolution as a basic operator, RS-CNN, a hierarchical architecture can be developed to achieve contextual shape-aware learning for point cloud analysis. Extensive experiments on challenging benchmarks across three tasks verify RS-CNN achieves the state of the arts.

# Motivation

[motivation]: ./images/motivation.jpg
![motivation]
<p align = 'center'>
<small>Left part: 3D Point cloud. Right part: Underlying shape formed by this point cloud.</small>
</p>

- The geometric relation among points is an explicit expression about the spatial layout of points, further discriminatively reflecting the underlying shape.

- CNN has demonstrated its powerful visual abstraction capability for 2D images that are in a regular grid format.

- Extending 2D grid CNN to 3D irregular configuration for point cloud analysis, by learning high-level geometric relation encoding from local to global.

# RS-Conv: Relation-Shape Convolution

[rsconv]: ./images/rsconv.jpg
![rsconv]
<p align = 'center'>
<small> Overview of our relation-shape convolution (RS-Conv). </small>
</p>

In this paper, we develop a hierarchical CNN-like architecture, _i.e._ RS-CNN, equipped with a novel learn-from-relation convolution operator called relation-shape convolution (RS-Conv). As illustrated in the figure, the key to RS-CNN is learning from relation.

Specifically:

- The convolutional weight for \\(x_{j}\\) is converted to \\({\bm{\mathrm w}}_{ij}\\), which learns a high-level mapping \\(\mathcal{M}\\) (Eq.~\eqref{Eq2:transform_relation}) on predefined geometric relation vector \\({\bm{\mathrm h}}_{ij}\\).

- In this way, the inductive convolutional representation \\(\sigma \big( \mathcal{A}(\{{\bm{\mathrm w}}_{ij} \cdot {\bm{\mathrm f}}_{x_j}, \hspace{0.1pt} \forall x_j\}) \big)\\) (Eq.~\eqref{Eq3:graph_relation}) can expressively reason the spatial layout of points, resulting in discriminative shape awareness.

- As in image CNN, further channel-raising mapping is conducted for a more powerful shape-aware representation.

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
