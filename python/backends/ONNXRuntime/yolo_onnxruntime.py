'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:25:29
Description: onnxruntime inference class for YOLO algorithm
'''


import onnxruntime
from backends.yolo import *
from backends.utils import *


'''
description: onnxruntime inference class for YOLO algorithm
'''
class YOLO_ONNXRuntime(YOLO):
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
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, os.cpu_count() // 2)
        if device_type == 'CPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        elif device_type == 'GPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CUDAExecutionProvider'])
        assert self.onnx_session is not None, 'onnx_session is None!'
        self.algo_type = algo_type
        self.model_type = model_type    
        self.inputs_name = []
        for node in self.onnx_session.get_inputs(): 
            self.inputs_name.append(node.name)
        self.inputs = {}
    
    '''
    description:    model infenence
    param {*} self  instance of class
    return {*}
    '''    
    def process(self) -> None:
        self.outputs = self.onnx_session.run(None, self.inputs)
