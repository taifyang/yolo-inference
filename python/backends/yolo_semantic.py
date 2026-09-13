'''
Author: taifyang
Date: 2026-09-10 23:38:07
LastEditTime: 2026-09-11 21:30:23
Description: class for YOLO semantic segmentation algorithm
'''


import numpy as np
from backends.yolo import *


'''
description: class for YOLO semantic segmentation algorithm
'''
class YOLO_Semantic(YOLO):
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
        YOLO.__init__(self)
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (1024, 1024)

    def draw_result(self, masks):
        image_copy = self.image.copy()   

        for mask in masks:
            image_copy[mask] = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
        result = (self.image*0.5 + image_copy*0.5).astype(np.uint8)       
        return result