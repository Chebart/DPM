from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import torch
import numpy as np
import cv2
from torchvision.ops import nms
import joblib
from scipy import ndimage
import xgboost as xgb

class DPM():

    """
    The implementation of DPM (Deformable part model) according to 
    original paper but the only difference is that I use xgboost instead of SVM.
    """

    def __init__(
        self, 
        image_h = 80,
        image_w = 80,
        step_x = 14, 
        step_y = 14, 
        orientations = 9,
        pix_per_cell_root = 16, 
        cells_per_block_root = 4, 
        pix_per_cell_part = 8, 
        cells_per_block_part = 2, 
        parts_count = 2, 
        part_w = (30,79), 
        part_h = (79,30),
        downscale = 2.0,
        conf_thresh = 0.35,
        IoU_thresh = 0.65,
        mode = "train"):

        # size of training imgs
        self.image_h = image_h
        self.image_w = image_w
        # step size of sliding window 
        self.step_x = step_x
        self.step_y = step_y
        # params for HOG descriptor of root and part filters 
        self.orientations = orientations
        self.pix_per_cell_root=pix_per_cell_root
        self.cells_per_block_root=cells_per_block_root
        self.pix_per_cell_part=pix_per_cell_part
        self.cells_per_block_part=cells_per_block_part
        # count of part filters
        self.parts_count = parts_count
        # size of part filters
        self.part_w = part_w 
        self.part_h = part_h
        # gaussian pyramid downscale
        self.downscale = downscale
        # thresholds for NMS
        self.conf_thresh = conf_thresh
        self.IoU_thresh = IoU_thresh
        # DPM mode
        self.mode = mode
        # get classifiers
        self.models = self.get_clfs()

    def get_clfs(self):
        models = {}
		
        if self.mode=="train":
            for i in range(self.parts_count+1):
                models[str(i)] = xgb.XGBClassifier( max_depth=12, subsample=0.33,
                                                    n_estimators=75, learning_rate = 0.2, 
                                                    colsample_bytree = 0.7, gamma = 1, 
                                                    reg_alpha = 30, reg_lambda = 0)
        else:
            for i in range(self.parts_count+1):
                models[str(i)] = joblib.load("./clfs/{}.sav".format(str(i)))

        return models
    
    def save_clfs(self):
        for i in range(self.parts_count+1):
            joblib.dump(self.models[str(i)], "./clfs/{}.sav".format(str(i))) 
    
    def initialize_part_filters(self):
        self.part_filters = joblib.load("./part_filters.npy")
    
    def save_part_filters(self, part_filters):
        joblib.dump(part_filters, "./part_filters.npy") 
    
    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in np.arange(0, image.shape[0]-windowSize[0], stepSize[1]):
            for x in np.arange(0, image.shape[1]-windowSize[1], stepSize[0]):
                yield (x, y, image[y: y + windowSize[0], x:x + windowSize[1]])

    def img_magnitude(self, img):
        # horizontal sobel filter
        dx = ndimage.sobel(img, 1)
        # vertical sobel filter
        dy = ndimage.sobel(img, 0)
        # calculate magnitude
        mag = np.hypot(dx, dy)
        # normalize
        mag *= 255.0 / np.max(mag)
        return mag

    def slice_magnitude(self,mag_map,rect):
        return np.sum(mag_map[rect[0]: rect[0] + rect[2],rect[1]:rect[1] + rect[3]],dtype=np.float64)
    
    def zero_magnitude(self,mag_map,rect):
        mag_map[rect[0]: rect[0] + rect[2],rect[1]:rect[1] + rect[3]] = 0

    def parts_of_image(self, img):
        mag_map = self.img_magnitude(img)
        vars = np.zeros((self.parts_count,3))

        for i in range(self.parts_count):
            for (x,y,_) in self.sliding_window(img, stepSize=(self.step_x,self.step_y), 
                                               windowSize=(self.part_h[i],self.part_w[i])):
                
                im_mag = self.slice_magnitude(mag_map,(y,x,self.part_h[i],self.part_w[i]))

                if vars[i][0] < im_mag:
                    vars[i][0] = im_mag
                    vars[i][1] = x
                    vars[i][2] = y

            self.zero_magnitude(mag_map,np.array((vars[i][2],vars[i][1],self.part_h[i],self.part_w[i])).astype(int))

        return np.array([[y,x] for _,x,y in vars])

    def collect_pathes_from_train(self,data):
        im_parts = np.array([self.parts_of_image(x) for x in data])
        return im_parts
 
    def compute_average(self,parts):
        median = np.median(parts,axis=1)
        result_parts = np.array([[np.sum(median[:i],axis=0)[0]/(i+1),
                                  np.sum(median[:i],axis=0)[1]/(i+1)] for i in range(len(median))])
        return result_parts

    def compute_part_filters(self, data):
        computed_parts = self.collect_pathes_from_train(data)
        res = self.compute_average(computed_parts)
        return res
            
    def process_part_filters(self,img):

        max_prob = {i:0 for i in range(1,self.parts_count + 1)}
        max_coord_point = {i:(self.part_h[i-1],self.part_w[i-1]) for i in range(1,self.parts_count + 1)}

        for i in range(self.parts_count):
            for (x,y,window) in self.sliding_window(img, stepSize=(self.step_x,self.step_y), 
                                                    windowSize=(self.part_h[i],self.part_w[i])):

                part_features = hog(window, orientations=self.orientations, 
                                pixels_per_cell=(self.pix_per_cell_part, self.pix_per_cell_part),
                                cells_per_block=(self.cells_per_block_part, self.cells_per_block_part),
                                block_norm= 'L2')

                probs = self.models[str(i+1)].predict_proba([part_features])
                if probs[0][1] > 0.75 and probs[0][1] > max_prob[i+1]:
                    max_coord_point[i+1] = (y,x)
                    max_prob[i+1] = probs[0][1]

        return max_prob, max_coord_point
	
    def get_filters_cost(self, best_coord, filters_nmb):
        return np.sqrt((np.mean(self.part_filters[filters_nmb - 1],axis=0)[0] - best_coord[0])**2 + 
                       (np.mean(self.part_filters[filters_nmb - 1],axis=0)[1] - best_coord[1])**2)/(self.parts_count)

    def process_frame(self,img):
        # constant
        summ_cost = 0

        # root branch cost
        root_features = hog(img, orientations=self.orientations, 
                            pixels_per_cell=(self.pix_per_cell_root, self.pix_per_cell_root),
                            cells_per_block=(self.cells_per_block_root, self.cells_per_block_root), block_norm= 'L2')

        if self.models["0"].predict([root_features]) == 1:
            summ_cost += 0.5

        # parts branches cost
        best_prob,best_coord = self.process_part_filters(img)
        key = max(best_prob, key=best_prob.get)

        for val in best_coord.keys():
            if best_prob[val]>0:
                filter_cost = self.get_filters_cost(best_coord[val],val)
                cost = 1.0/(1 + filter_cost)
                summ_cost += cost

        return [(summ_cost-self.parts_count*0.02)/((self.parts_count+0.5)-self.parts_count*0.02),
                best_coord[key],self.part_w[key-1],self.part_h[key-1]]
                
    def process_image(self, img, name):

        # convert RGB img to grayscale img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = []

        # loop over each layer of the image
        for resized in pyramid_gaussian(img, downscale=self.downscale):
            # loop over the sliding window for each layer of the pyramid
            for (x,y,window) in self.sliding_window(resized, stepSize=(self.step_x,self.step_y), 
                                                    windowSize=(self.image_h,self.image_w)):
                # Process part of an image
                res = self.process_frame(window)

                if res[0] > self.conf_thresh:
                    scale = img.shape[0]//resized.shape[0]
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    print("Scale ->  {} | Confidence Score {}".format(scale,res[0]))
                    print("Width and Height of bounding box -> ({}, {}) \n".format(res[2],res[3]))
                    detections.append([(x+res[1][1])*scale,(y+res[1][0])*scale,res[0],res[2]*scale,res[3]*scale])


        # get bounding boxes and scores from detections
        rects = torch.Tensor([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        scores = torch.Tensor([score for (_, _, score, _, _) in detections])

        # if rects is not empty
        if rects.numel():
            # indices of the elements that have been kept by NMS
            idxs = nms(rects,scores,self.IoU_thresh)
            # get filtered bounding boxes 
            rects=rects[idxs]
            scores=scores[idxs]
            # draw bounding boxes
            for j in range(len(rects)):
                    cv2.rectangle(img, (int(rects[j][0].item()), int(rects[j][1].item())),
                                (int(rects[j][2].item()), int(rects[j][3].item())), (255,255,255), 1)

        # In my case I have binary classification: ship - 1
        labels=torch.ones(len(rects),dtype=torch.int)
        # save results of detection
        cv2.imwrite('./result/'+ name + '.jpg',img)
        return {"boxes":rects,"scores":scores,"labels":labels}