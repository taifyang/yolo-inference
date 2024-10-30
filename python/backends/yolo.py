'''
Author: taifyang 
Date: 2024-06-12 22:23:07
LastEditors: taifyang
LastEditTime: 2024-10-24 22:15:54
FilePath: \python\backends\yolo.py
Description: YOLO algorithm interface class
'''


import os
import cv2
import time
import backends
    

'''
description: YOLO algorithm interface class
'''
class YOLO:  
    '''
    description:    construction method
    param {*} self  instance of class
    return {*}
    '''    
    def __init__(self) -> None:
        super().__init__()
        self.class_num = 80		            #类别数量
        self.score_threshold = 0.2      	#得分阈值
        self.nms_threshold = 0.5        	#NMS阈值
        self.confidence_threshold = 0.2 	#置信度阈值    
        self.input_shape = (640, 640)   	#输入图像尺寸

    '''
    description:    task map
    param {*} self  instance of class
    return {*}      algorithm class instance
    '''    
    def task_map(self):
        map = {}
        try:
            import onnxruntime
            map['ONNXRuntime'] = {
                'Classify':backends.ONNXRuntime.YOLO_ONNXRuntime_Classify,
                'Detect':backends.ONNXRuntime.YOLO_ONNXRuntime_Detect,
                'Segment':backends.ONNXRuntime.YOLO_ONNXRuntime_Segment,
            }
        except:
               pass
        
        try:
            import cv2   
            map['OpenCV'] =  {
                'Classify':backends.OpenCV.YOLO_OpenCV_Classify,
                'Detect':backends.OpenCV.YOLO_OpenCV_Detect,
                'Segment':backends.OpenCV.YOLO_OpenCV_Segment,
            }
        except:
            pass
                             
        try:
            import openvino
            map['OpenVINO'] = {
                'Classify':backends.OpenVINO.YOLO_OpenVINO_Classify,
                'Detect':backends.OpenVINO.YOLO_OpenVINO_Detect,
                'Segment':backends.OpenVINO.YOLO_OpenVINO_Segment,
            }
        except:
            pass
        
        try:
            import torch
            map['PyTorch'] =  {
                'Classify':backends.PyTorch.YOLO_PyTorch_Classify,
                'Detect':backends.PyTorch.YOLO_PyTorch_Detect,
                'Segment':backends.PyTorch.YOLO_PyTorch_Segment,
            }
        except:
            pass
        
        try:
            import tensorrt
            map['TensorRT'] = {
                'Classify':backends.TensorRT.YOLO_TensorRT_Classify,
                'Detect':backends.TensorRT.YOLO_TensorRT_Detect,
                'Segment':backends.TensorRT.YOLO_TensorRT_Segment,
            }
        except:
            pass

        return map
    
    '''
    description:                inference interface
    param {*} self              instance of class
    param {str} input_path      input path
    param {str} output_path     output path
    param {bool} save_result    save result
    param {bool} show_result    show result
    return {*}
    '''    
    def infer(self, input_path:str, output_path:str, save_result:bool, show_result:bool) -> None:
        assert os.path.exists(input_path), 'input not exists!'
        self.draw_result = save_result or show_result
        if input_path.endswith('.bmp') or input_path.endswith('.jpg') or input_path.endswith('.png'):
            self.image = cv2.imread(input_path)
            self.result = self.image.copy()
        
            self.pre_process()
            self.process()
            self.post_process()
            
            start = time.perf_counter()
            for i in range(10):
                self.pre_process()
                self.process()
                self.post_process()
            end = time.perf_counter()
            print('avg cost:', (end-start)*100, 'ms')  
            
            if save_result and output_path!='':
                cv2.imwrite(output_path, self.result)
            if show_result:
                cv2.imshow('result', self.result)
                cv2.waitKey(0)
        elif input_path.endswith('.mp4'):
            cap = cv2.VideoCapture(input_path)
            start = time.perf_counter()
            if save_result and output_path!='':
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
                    cv2.imshow('result', self.result)
                    cv2.waitKey(1)
                if save_result and output_path!='':
                    wri.write(self.result)
            end = time.perf_counter()
            print((end-start)*1000, 'ms')                         
            
    