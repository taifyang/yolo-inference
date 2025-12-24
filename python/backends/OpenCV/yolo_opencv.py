'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:26:15
Description: opencv inference class for YOLO algorithm
'''

import cv2
from backends.yolo import *
from backends.utils import *


'''
description: opencv inference class for YOLO algorithm
'''
class YOLO_OpenCV(YOLO):
    '''
    description:            construction method
    param {*} self          instance of class
    param {str} algo_type   algorithm type
    param {str} device_type device type
    param {str} model_type  model type
    param {str} model_path  model path
    return {*}
    '''     
    def __init__(self, algo_type:str, device_type:str, model_type:str, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), 'model not exists!'
        assert device_type in ['CPU', 'GPU'], 'unsupported device type!'
        assert model_type in ['FP32', 'FP16'], 'unsupported model type!'
        self.net = cv2.dnn.readNet(model_path)
        self.algo_type = algo_type
        if device_type == 'CPU':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device_type == 'GPU':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            if model_type == 'FP32':
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            elif model_type == 'FP16':
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        assert self.net.empty() == False, 'model load failed!'
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
