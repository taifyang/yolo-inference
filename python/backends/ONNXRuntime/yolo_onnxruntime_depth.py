'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 23:12:33
Description: onnxruntime inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.yolo_depth import *
from backends.ONNXRuntime.yolo_onnxruntime import *
            

'''
description: onnxruntime inference class for the YOLO depth estimation algorithm
'''      
class YOLO_ONNXRuntime_Depth(YOLO_ONNXRuntime, YOLO_Depth):
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
        YOLO_ONNXRuntime.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Depth.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        if self.model_type == 'FP32' or self.model_type == 'INT8':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == 'FP16':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        self.inputs[self.inputs_name[0]] = input
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)
        depth = scale_masks(output, self.inputs_shape, self.image.shape)
        if self.render_result:
            self.result = self.draw_result(depth)