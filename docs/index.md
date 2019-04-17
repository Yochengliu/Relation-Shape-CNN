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

<div align="center">
    <img src="images/partseg.jpg" width="80%" height ="80%" alt="partseg.jpg" />
</div>
<p align = 'center'>
    <small>Segmentation examples on ShapeNet part benchmark. Although the part shapes implied in irregular points are extremely diverse and they may be very confusing to recognize, our RS-CNN can also segment them out with decent accuracy.</small>
</p>

<h1 align = "center">Abstract</h1> 

Point cloud analysis is very challenging, as the shape implied in irregular points is difficult to capture. In this paper, we propose RS-CNN, namely, Relation-Shape Convolutional Neural Network, which extends regular grid CNN to irregular configuration for point cloud analysis. ___The key to RS-CNN is learning from relation___, i.e., the geometric topology constraint among points. Specifically, the convolutional weight for local point set is forced to ___learn a high-level relation expression from predefined geometric priors___, between a sampled point from this point set and the others. In this way, an inductive local representation with ___explicit reasoning about the spatial layout of points___ can be obtained, which leads to much shape awareness and robustness. With this convolution as a basic operator, RS-CNN, a hierarchical architecture can be developed to achieve contextual shape-aware learning for point cloud analysis. Extensive experiments on challenging benchmarks across three tasks verify RS-CNN achieves the state of the arts.

<h1 align = "center">Motivation</h1> 

<div align="center">
    <img src="images/motivation.jpg" width="60%" height ="60%" alt="motivation.jpg" />
</div>
<p align = 'center'>
    <small>Left part: 3D Point cloud. Right part: Underlying shape formed by this point cloud.</small>
</p>

- The geometric relation among points is an explicit expression about the spatial layout of points, further discriminatively reflecting the underlying shape.

- CNN has demonstrated its powerful visual abstraction capability for 2D images that are in the format of a regular grid.

- Can we extend 2D grid CNN to 3D irregular configuration for point cloud analysis, by learning high-level geometric relation encoding for discriminative shape awareness?

<h1 align = "center">RS-Conv: Relation-Shape Convolution</h1>

[rsconv]: ./images/rsconv.jpg
![rsconv]
<p align = 'center'>
<small> Overview of our relation-shape convolution (RS-Conv). </small>
</p>

In this paper, we develop a hierarchical CNN-like architecture, _i.e._ RS-CNN, equipped with a novel learn-from-relation convolution operator called relation-shape convolution (RS-Conv). As illustrated in the figure, the key to RS-CNN is learning from relation.

To be specific:

- The convolutional weight <img src="maths/w_strong.png" align="center" border="0" weight="24" height="16" alt="{\bm{\mathrm w}}_j" /> for <img src="maths/xj.png" align="center" border="0" alt="x_{j}" width="21" height="16" /> is converted to <img src="maths/wij.png" align="center" border="0" alt="{\bm{\mathrm w}}_{ij}" width="28" height="16" />, which learns a high-level mapping <img src="maths/m.png" align="center" border="0" alt="\mathcal{M}" width="28" height="16" /> (<img src="maths/wijm.png" align="center" border="0" alt="{\bm{\mathrm w}}_{ij}=\mathcal{M}({\bm{\mathrm h}}_{ij})" width="90" height="16" />) on predefined geometric relation vector <img src="maths/hij.png" align="center" border="0" alt="{\bm{\mathrm h}}_{ij}" width="20" height="16" />.

- In this way, the inductive convolutional representation <img src="maths/conv.png" align="center" border="0" weight="154" height="22"  alt="\sigma \big( \mathcal{A}(\{{\bm{\mathrm w}}_{ij} \cdot {\bm{\mathrm f}}_{x_j}, \hspace{0.1pt} \forall x_j\}) \big)"/> can expressively reason the spatial layout of points, resulting in discriminative shape awareness.

- As in image CNN, further channel-raising mapping is conducted for a more powerful shape-aware representation.

<h1 align = "center">Revisiting 2D Grid Convolution</h1>

<div align="center">
    <img src="images/2dconv.jpg" width="50%" height ="50%" alt="2dconv.jpg" />
</div>
<p align = 'center'>
<small> Illustration of 2D grid convolution with a kernel of 3 x 3. </small>
</p>

- The convolutional weight <img src="maths/swj.png" align="center" border="0" alt="w_{j}" width="31" height="21" /> for <img src="maths/xj.png" align="center" border="0" alt="x_{j}" width="27" height="21" /> always implies a fixed positional relation between <img src="maths/xi.png" align="center" border="0" alt="x_{i}" width="27" height="19" /> and its neighbor <img src="maths/xj.png" align="center" border="0" alt="x_{j}" width="27" height="21" /> in the regular grid. That is, <img src="maths/swj.png" align="center" border="0" alt="w_{j}" width="31" height="21" /> is actually constrained to encode one kind of regular grid relation in the learning process.

- Therefore, our RS-Conv with relation learning is more general and can be applied to model 2D grid spatial relationship.

<h1 align = "center">Experiment</h1>

### Shape Classification on ModelNet40 Benchmark

<div align="center">
    <img src="images/cls.jpg" width="45%" height ="45%" alt="cls.jpg" />
</div>
<p align = 'center'>
<small> Shape classification results (%). Our RS-CNN outperforms the state of the arts with only <img src="maths/xyz.png" align="center" border="0" alt="\mathrm{xyz}" width="32" height="19" /> as the input features. </small>
</p>

### Normal Estimation

<div align="center">
    <img src="images/normal.jpg" width="65%" height ="65%" alt="normal.jpg" />
</div>
<p align = 'center'>
<small> Normal estimation examples. For clearness, we only show predictions with angle less than 30 degree in blue, and angle greater than 90 degree in red between the ground truth normals. </small>
</p>

### Relation Definition

<div align="center">
    <img src="images/relation.jpg" width="60%" height ="60%" alt="relation.jpg" />
</div>
<p align = 'center'>
<small> The results (%) of five intuitive low-level relation. Model A applies only 3D Euclidean distance; Model B adds the coordinates difference to model A; Model C adds the coordinates of two points to model B; Model D utilizes the normals of two points and their cosine distance; Model E projects 3D points onto a 2D plane of XY, XZ and YZ. </small>
</p>

### Robustness 

<div align="center">
    <img src="images/density.jpg" width="80%" height ="80%" alt="density.jpg" />
</div>
<p align = 'center'>
<small> Robustness to sampling density. Left part: Point cloud with random point dropout. Right part: Test results of using sparser points as the input to a model trained with 1024 points. </small>
</p>

<div align="center">
    <img src="images/rotation.jpg" width="70%" height ="70%" alt="rotation.jpg" />
</div>
<p align = 'center'>
<small> Robustness to point permutation and rigid transformation (%). During testing, we perform random permutation (perm.) of points, add a small translation of 0.2 and rotate the input point cloud by 90 degree and 180 degree. </small>
</p>

<h1 align = "center">Visualization and Complexity</h1>

### Visualization

<div align="center">
    <img src="images/visualization.jpg" width="70%" height ="70%" alt="visualization.jpg" />
</div>
<p align = 'center'>
<small> The features learned by the first layer mostly respond to edges, corners and arcs, while the ones in the second layer capture more semantical shape parts like airfoils and heads. </small>
</p>

### Complexity

<div align="center">
    <img src="images/complexity.jpg" width="50%" height ="50%" alt="complexity.jpg" />
</div>
<p align = 'center'>
<small> Complexity of RS-CNN in point cloud classification. </small>
</p>

# Code

Please refer to the [GitHub repository](https://github.com/Yochengliu/Relation-Shape-CNN) for more details. 

# Publication

Yongcheng Liu, Bin Fan, Shiming Xiang and Chunhong Pan, "Relation-Shape Convolutional Neural Network for Point Cloud Analysis", in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. [[arXiv](https://arxiv.org/abs/1904.07601)]

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
