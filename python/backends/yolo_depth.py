'''
Author: taifyang
Date: 2026-09-10 23:38:07
LastEditTime: 2026-09-11 23:22:44
Description: class for YOLO depth estimation algorithm
'''


import numpy as np
from backends.yolo import *


'''
description: class for YOLO depth estimation algorithm
'''
class YOLO_Depth(YOLO):
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
        self.inputs_shape = (768, 768)

    def draw_result(self, image):
        depth = np.clip(image * 1000, 0, 65535).astype(np.uint16)
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return depth