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
    <img src="images/partseg.jpg" width="90%" height ="90%" alt="partseg.jpg" />
</div>
<p align = 'center'>
    <small>Segmentation examples on ShapeNet part benchmark. Although the part shapes implied in irregular points are extremely diverse and they may be very confusing to recognize, our RS-CNN can also segment them out with decent accuracy.</small>
</p>

<h2 align = "center">Abstract</h2> 

Point cloud analysis is very challenging, as the shape implied in irregular points is difficult to capture. In this paper, we propose RS-CNN, namely, Relation-Shape Convolutional Neural Network, which extends regular grid CNN to irregular configuration for point cloud analysis. ___The key to RS-CNN is learning from relation___, i.e., the geometric topology constraint among points. Specifically, the convolutional weight for local point set is forced to ___learn a high-level relation expression from predefined geometric priors___, between a sampled point from this point set and the others. In this way, an inductive local representation with ___explicit reasoning about the spatial layout of points___ can be obtained, which leads to much shape awareness and robustness. With this convolution as a basic operator, RS-CNN, a hierarchical architecture can be developed to achieve contextual shape-aware learning for point cloud analysis. Extensive experiments on challenging benchmarks across three tasks verify RS-CNN achieves the state of the arts.

<h2 align = "center">Motivation</h2> 

<div align="center">
    <img src="images/motivation.jpg" width="60%" height ="60%" alt="motivation.jpg" />
</div>
<p align = 'center'>
    <small>Left part: 3D Point cloud. Right part: Underlying shape formed by this point cloud.</small>
</p>

- The geometric relation among points is an explicit expression about the spatial layout of points, further discriminatively reflecting the underlying shape.

- CNN has demonstrated its powerful visual abstraction capability for 2D images that are in the format of a regular grid.

- Can we extend 2D grid CNN to 3D irregular configuration for point cloud analysis, by learning high-level geometric relation encoding for discriminative shape awareness?

<h2 align = "center">RS-Conv: Relation-Shape Convolution</h2>

[rsconv]: ./images/rsconv.jpg
![rsconv]
<p align = 'center'>
<small> Overview of our relation-shape convolution (RS-Conv). </small>
</p>

In this paper, we develop a hierarchical CNN-like architecture, _i.e._ RS-CNN, equipped with a novel learn-from-relation convolution operator called relation-shape convolution (RS-Conv). As illustrated in the figure, the key to RS-CNN is learning from relation.

To be specific:

- The convolutional weight <img src="maths/w_strong.png" align="center" border="0" weight="50%" height="50%" alt="{\bm{\mathrm w}}_j" /> for <img src="http://www.sciweavers.org/tex2img.php?eq=x_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="x_{j}" width="27" height="21" /> is converted to <img src="http://www.sciweavers.org/tex2img.php?eq=%7B%5Cbm%7B%5Cmathrm%20w%7D%7D_%7Bij%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="{\bm{\mathrm w}}_{ij}" width="37" height="21" />, which learns a high-level mapping <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmathcal%7BM%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="\mathcal{M}" width="37" height="21" /> (<img src="http://www.sciweavers.org/tex2img.php?eq=%7B%5Cbm%7B%5Cmathrm%20w%7D%7D_%7Bij%7D%3D%5Cmathcal%7BM%7D%28%7B%5Cbm%7B%5Cmathrm%20h%7D%7D_%7Bij%7D%29&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="{\bm{\mathrm w}}_{ij}=\mathcal{M}({\bm{\mathrm h}}_{ij})" width="148" height="29" />) on predefined geometric relation vector <img src="http://www.sciweavers.org/tex2img.php?eq=%7B%5Cbm%7B%5Cmathrm%20h%7D%7D_%7Bij%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="{\bm{\mathrm h}}_{ij}" width="33" height="27" />.

- In this way, the inductive convolutional representation <img src="maths/conv.png" align="center" border="0" weight="50%" height="50%"  alt=""/> can expressively reason the spatial layout of points, resulting in discriminative shape awareness.

- As in image CNN, further channel-raising mapping is conducted for a more powerful shape-aware representation.

<h2 align = "center">Revisiting 2D Grid Convolution</h2>

<div align="center">
    <img src="images/2dconv.jpg" width="60%" height ="60%" alt="2dconv.jpg" />
</div>
<p align = 'center'>
<small> Illustration of 2D grid convolution with a kernel of 3 x 3. </small>
</p>

- The convolutional weight <img src="http://www.sciweavers.org/tex2img.php?eq=w_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="w_{j}" width="31" height="21" /> for <img src="http://www.sciweavers.org/tex2img.php?eq=x_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="x_{j}" width="27" height="21" /> always implies a fixed positional relation between <img src="http://www.sciweavers.org/tex2img.php?eq=x_%7Bi%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="x_{i}" width="27" height="19" /> and its neighbor <img src="http://www.sciweavers.org/tex2img.php?eq=x_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="x_{j}" width="27" height="21" /> in the regular grid. That is, <img src="http://www.sciweavers.org/tex2img.php?eq=w_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="w_{j}" width="31" height="21" /> is actually constrained to encode one kind of regular grid relation in the learning process.

- Therefore, our RS-Conv with relation learning is more general and can be applied to model 2D grid spatial relationship.

<h2 align = "center">Experiment</h2>

### Shape Classification on ModelNet40 Benchmark

[cls]: ./images/cls.jpg
![cls]
<p align = 'center'>
<small> Shape classification results (%). Our RS-CNN outperforms the state of the arts with only <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmathrm%7Bxyz%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="\mathrm{xyz}" width="42" height="19" /> as the input features. </small>
</p>

### Normal Estimation

[normal]: ./images/normal.jpg
![normal]
<p align = 'center'>
<small> Normal estimation examples. For clearness, we only show predictions with angle less than 30 degree in blue, and angle greater than 90 degree in red between the ground truth normals. </small>
</p>

### Relation Definition

[relation]: ./images/relation.jpg
![relation]
<p align = 'center'>
<small> The results (%) of five intuitive low-level relation. Model A applies only 3D Euclidean distance; Model B adds the coordinates difference to model A; Model C adds the coordinates of two points to model B; Model D utilizes the normals of two points and their cosine distance; Model E projects 3D points onto a 2D plane of XY, XZ and YZ. </small>
</p>

### Robustness 

[density]: ./images/density.jpg
![density]
<p align = 'center'>
<small> Robustness to sampling density. Left part: Point cloud with random point dropout. Right part: Test results of using sparser points as the input to a model trained with 1024 points. </small>
</p>

[rotation]: ./images/rotation.jpg
![rotation]
<p align = 'center'>
<small> Robustness to point permutation and rigid transformation (%). During testing, we perform random permutation (perm.) of points, add a small translation of 0.2 and rotate the input point cloud by 90 degree and 180 degree. </small>
</p>

<h2 align = "center">Visualization and Complexity</h2>

### visualization
[Visualization]: ./images/visualization.jpg
![Visualization]
<p align = 'center'>
<small> The features learned by the first layer mostly respond to edges, corners and arcs, while the ones in the second layer capture more semantical shape parts like airfoils and heads. </small>
</p>

### Complexity
[complexity]: ./images/complexity.jpg
![complexity]
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
