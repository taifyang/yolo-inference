import cv2
import numpy as np
from enum import Enum
from abc import ABC, abstractclassmethod

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] #coco80类别   
    
input_shape = (640, 640) 

score_threshold = 0.2  

nms_threshold = 0.5

confidence_threshold = 0.2  

class Device_Type(Enum):
    CPU = 0
    GPU = 1


class YOLOv5(ABC):
    def infer(self, image_path:str) -> None:
        self.image = cv2.imread(image_path)
        self.result = self.image.copy()
        self.pre_process()
        self.process()
        self.post_process()
        cv2.imwrite("result.jpg", self.result)
        cv2.imshow("result", self.result)
        cv2.waitKey(0)
    
    @abstractclassmethod
    def pre_process(self) -> None:
        pass
    
    @abstractclassmethod
    def process(self) -> None:
        pass    
    
    @abstractclassmethod
    def post_process(self) -> None:
        pass
     
    