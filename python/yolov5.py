import os
import cv2
import time
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

class Model_Type(Enum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    

class YOLOv5(ABC):  
    def infer(self, file_path:str) -> None:
        assert os.path.exists(file_path)
        if file_path[-4:] == ".bmp" or file_path[-4:] == ".jpg" or file_path[-4:] == ".png":
            self.image = cv2.imread(file_path)
            self.result = self.image.copy()
            self.pre_process()
            self.process()
            self.post_process()
            cv2.imwrite("result.jpg", self.result)
            cv2.imshow("result", self.result)
            cv2.waitKey(0)
        elif file_path[-4:] == ".mp4":
            cap = cv2.VideoCapture(file_path)
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #wri = cv2.VideoWriter('result.avi', fourcc, 30.0, (1280,720))
            while cv2.waitKey(1) < 0:
                start = time.time()
                ret, self.image  = cap.read()
                if not ret:
                    break
                self.result = self.image.copy()
                self.pre_process()
                self.process()
                self.post_process()
                #cv2.imshow("result", self.result)
                #wri.write(self.result)
                end = time.time()
                print((end-start)*1000, "ms")                         
            
    @abstractclassmethod
    def pre_process(self) -> None:
        pass
    
    @abstractclassmethod
    def process(self) -> None:
        pass    
    
    @abstractclassmethod
    def post_process(self) -> None:
        pass
     
    