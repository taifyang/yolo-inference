'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:26:56
Description: openvino inference class for YOLO algorithm
'''

import openvino as ov
from backends.yolo import *
from backends.utils import *

'''
description: openvino inference class for YOLO algorithm
'''
class YOLO_OpenVINO(YOLO):
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
        try:
            from openvino.runtime import Core
            core = Core()
        except:
            core = ov.Core()
        model  = core.read_model(model_path)
        self.algo_type = algo_type
        self.compiled_model = core.compile_model(model, device_name='GPU' if device_type=='GPU' else 'CPU')
        assert self.compiled_model, 'compile model failed!'
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.compiled_model({0: self.inputs})

