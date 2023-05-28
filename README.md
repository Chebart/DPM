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

## Explain class variables

## Results
