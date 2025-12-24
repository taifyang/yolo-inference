'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:34:35
Description: pytorch inference class for YOLO algorithm
'''


import torch
import torchvision
import torch.nn.functional as F
from backends.yolo import *
from backends.utils import *


'''
description: pytorch inference class for YOLO algorithm
'''
class YOLO_PyTorch(YOLO):
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
        self.net = torch.jit.load(model_path)
        assert self.net is not None, 'load model failed!'
        self.algo_type = algo_type
        self.device_type = device_type
        self.model_type = model_type
        if self.device_type == 'GPU':
            self.net = self.net.cuda()
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.net(self.inputs)
