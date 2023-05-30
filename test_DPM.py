from custom_DPM import DPM
import torch
import os
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from xml.etree import ElementTree as et
import cv2
from pprint import pprint

# Function for parsing .xml files
def get_true_bboxes():
    labels = [i for i in os.listdir("./test") if '.xml' in i]
    classes = ['__background__', 'ship']
    gr_true_bboxxes = []
    # height and width of the input image
    image_width = 768
    image_height = 768
    # height and width of the output image
    out_width = 384
    out_height = 384
    
    for lab in labels:
        annot_file_path = os.path.join("./test", lab)
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        boxes = []
        labs = []
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object  to get the label index
            labs.append(classes.index(member.find('name').text))
            # left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the params
            xmin_final = (xmin/image_width)*out_width
            xmax_final = (xmax/image_width)*out_width
            ymin_final = (ymin/image_height)*out_height
            ymax_final = (ymax/image_height)*out_height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels to tensor
        labs = torch.as_tensor(labs, dtype=torch.int64)
        # add to the list
        gr_true_bboxxes.append({"boxes":boxes,"labels":labs})

    return gr_true_bboxxes


if __name__ == '__main__':
    # get ground truth boxes
    gr_tr_bboxes = get_true_bboxes()

    # initialize DPM
    dpm = DPM(mode="eval")
    # load DPM part filters
    dpm.initialize_part_filters()

    # list of detections
    all_detections = []
    # get test images
    imgs = [i for i in os.listdir("./test") if '.jpg' in i]

    # process test images
    start_time = time.time()
    for i in range(len(imgs)):
        img = cv2.imread("./test/"+imgs[i])
        # we divide width and height by 2 as for true bboxes
        img = cv2.resize(img, dsize=(img.shape[0]//2, img.shape[1]//2), interpolation=cv2.INTER_CUBIC)
        cur_detection = dpm.process_image(img, str(i+1))
        all_detections.append(cur_detection)

    # evaluate metric
    metric = MeanAveragePrecision()
    metric.update(all_detections, gr_tr_bboxes)
    pprint(metric.compute())
    print(metric)
    # measure work time for 1 image
    end_time = time.time()
    print("{:.2f} seconds per image".format((end_time-start_time)/len(imgs)))