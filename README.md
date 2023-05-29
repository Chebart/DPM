# Deformable Part-based Model(DPM)
<p align="center">
  <img src="https://github.com/Chebart/DPM/assets/88379173/e845c79b-4fe4-4021-bb09-b1fbe166d73d">
</p>

Implementation of Deformable Part-based Model according to original paper in Python.

Object recognition is one of the fundamental challenges in computer vision. Before the widespread use of neural networks, 
the recognition and localization categories of objects such as people or cars in static images was problematic since they
can vary greatly in appearance. Variations arise not only from changes in illumination and viewpoint, but also
due to non-rigid deformations, and intraclass variability in shape and other visual properties. For example, people wear 
different clothes and take a variety of poses while cars come in a various shapes and colors. The problem of detecting an 
object can be considered as a problem of detecting its parts. Thus, DPM is an object detection system which based on deformable models 
that represent objects using local part templates and geometric constraints on the locations of parts. 

In this implementation, the only thing that differs from the original paper is that xgboost is used instead of SVM.


## Useful links

1. [Dalal N., Triggs B. Histograms of oriented gradients for human detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
		//2005 IEEE Computer Society Conference on Computer Vision and
		Pattern Recognition (CVPR'05). – IEEE, 2005. – Т. 1. – С. 886-893.

2. [Felzenszwalb P. et al. Visual object detection with deformable part
		models](https://www.islab.ulsan.ac.kr/files/announcement/449/Visual%20Object%20Detection%20with%20Deformable%20Part%20Models%20ACM2013.pdf) //Communications of the ACM. – 2013. – Т. 56. – №. 9. – С.97-105.
	
3. [Felzenszwalb P., McAllester D., Ramanan D. A discriminatively trained,
		multiscale, deformable part model](https://cs.brown.edu/people/pfelzens/papers/latent.pdf) //Computer Vision and Pattern
		Recognition, 2008. CVPR 2008. IEEE Conference on. – IEEE, 2008. – С.1-8.
	
4. [Felzenszwalb P. F. et al. Object detection with discriminatively trained
		part-based models](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf) //IEEE transactions on pattern analysis and machine
		intelligence. – 2010. – Т. 32. – №. 9. – С. 1627-1645.
    
5. [Q&A of the Deformable Part Model by Philipp Krähenbühl](http://vision.stanford.edu/teaching/cs231b_spring1213/slides/detection_QA.pdf) // presentation from Machine Learning Course, 2013

## Python packages
```
numpy==1.23.1
scipy==1.10.0
pytorch==1.13.1+cu117
cv2==4.7.0.68
torchvision==0.14.1+cu117
joblib==1.2.0
xgboost==1.7.5
scikit-image==0.18.1
torchmetrics==0.11.4
```

## Dataset
The subset of images from [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data) dataset was used to train and evaluate DPM.

The trainig dataset consists of 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification. Images were derived from PlanetScope full-frame visual 
scene products, which are orthorectified to a 3-meter pixel size. The dataset is distributed as a JSON formatted text file shipsnet.json. You can download it [here](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery).

<p align="center">
  <img src="https://github.com/Chebart/DPM/assets/88379173/2d8c9035-62b3-452f-bb83-f07c2895c5cf">
</p>

For evaluation we used 50 random 768x768 RGB images from the original dataset. They look something like this:

<p align="center">
  <img width = 200 height = 200 src="https://github.com/Chebart/DPM/assets/88379173/71d44840-e869-41eb-a264-870bbc7847be">
  <img width = 200 height = 200 src="https://github.com/Chebart/DPM/assets/88379173/acf3db19-7b30-47c9-8fcc-2169cd38072b">
  <img width = 200 height = 200 src="https://github.com/Chebart/DPM/assets/88379173/4f3c1e79-3441-4936-8680-8de86788d50f">
  <img width = 200 height = 200 src="https://github.com/Chebart/DPM/assets/88379173/a10d0cba-a956-4914-91fd-1309faba5b7d">
</p>


## Explain class variables
<img align="right" src="https://github.com/Chebart/DPM/assets/88379173/843487ad-e788-47cb-9a6b-16492898b641">

- **self.image_h** - training image and sliding window height.
- **self.image_w** - training image and sliding window width.
- **self.step_x** - step size of sliding window along the x-axis.
- **self.step_y** - step size of sliding window along the y-axis.
- **self.orientations** - number of orientation bins (for HOG).
- **self.pix_per_cell_root** - size of a cell in pixels in root filter (for HOG).
- **self.cells_per_block_root** - number of cells in root filter block (for HOG).
- **self.pix_per_cell_part** - size of a cell in pixels in part filter (for HOG).
- **self.cells_per_block_part** - number of cells in part filter block (for HOG).
- **self.parts_count** - count of part filters.
- **self.part_w** - list of length self.parts_count which stores the part filters widths. Must be < self.image_w.
- **self.part_h** - list of length self.parts_count which stores the part filters heights. Must be < self.image_h.
- **self.downscale** - image downscale factor on each pyramid layer.
- **self.conf_thresh** - score that the model will consider the prediction to be a true.
- **self.IoU_thresh** - score to evaluate the overlap between predicted bounding boxes and ground truth bounding boxes to be considered a true positive.
- **self.mode** - flag which determines whether to train models or evaluate.
- **self.models** - dict of latent variable models.

## Results
