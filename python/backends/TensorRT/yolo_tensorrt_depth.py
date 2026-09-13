'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-13 21:33:14
Description: tensorrt inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.yolo_depth import *
from backends.TensorRT.yolo_tensorrt import *


'''
description: tensorrt inference class for the YOLO depth estimation algorithm
'''             
class YOLO_TensorRT_Depth(YOLO_TensorRT, YOLO_Depth):
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
        YOLO_TensorRT.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Depth.__init__(self, algo_type, device_type, model_type, model_path)
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output0_ptr = self.output0_device.data.ptr
               
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''        
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape, use_cupy=True)
        input = cupy.transpose(input[:, :, ::-1], (2, 0, 1))
        input = input.astype(cupy.float32) / 255.0
        self.input_ptr = input.data.ptr
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.context.execute_v2(bindings=[self.input_ptr, self.output0_ptr])
        self.output0_host = cupy.asnumpy(self.output0_device) 

    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''            
    def post_process(self) -> None:
        output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
        depth = scale_masks(output, self.inputs_shape, self.image.shape)
        if self.render_result:
            self.result = self.draw_result(depth)
 