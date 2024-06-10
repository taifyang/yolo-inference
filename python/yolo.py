import os
import cv2
import time
from pathlib import Path
from enum import Enum
from abc import ABC, abstractclassmethod
 
    
class Algo_Type(Enum):
    YOLOv5 = 0
    YOLOv8 = 1
    
class Task_Type(Enum):
    Classification = 0
    Detection = 1
    Segmentation = 2

class Device_Type(Enum):
    CPU = 0
    GPU = 1

class Model_Type(Enum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    

class YOLO(ABC):  
    def __init__(self) -> None:
        super().__init__()
        self.score_threshold = 0.2
        self.nms_threshold = 0.5
        self.confidence_threshold = 0.2  
        self.input_shape = (640, 640) 
    
    def infer(self, input_path:str, output_path:str, show_result:bool, save_result:bool) -> None:
        assert os.path.exists(input_path), "input not exists!"
        if input_path.endswith(".bmp") or input_path.endswith(".jpg") or input_path.endswith(".png"):
            self.image = cv2.imread(input_path)
            self.pre_process()
            self.process()
            self.post_process()
            if save_result and output_path!="":
                cv2.imwrite(output_path, self.result)
            if show_result:
                cv2.imshow("result", self.result)
                cv2.waitKey(0)
        elif input_path.endswith(".mp4"):
            cap = cv2.VideoCapture(input_path)
            start = time.time()
            if save_result and output_path!="":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                wri = cv2.VideoWriter(output_path, fourcc, 30.0, (1280,720))
            while True:
                ret, self.image  = cap.read()
                if not ret:
                    break
                self.result = self.image.copy()
                self.pre_process()
                self.process()
                self.post_process()
                if show_result:
                    cv2.imshow("result", self.result)
                    cv2.waitKey(1)
                if save_result and output_path!="":
                    wri.write(self.result)
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
     
    