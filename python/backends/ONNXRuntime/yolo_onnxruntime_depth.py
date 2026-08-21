'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-08-03 22:19:15
Description: onnxruntime inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.ONNXRuntime.yolo_onnxruntime import *
            

'''
description: onnxruntime inference class for the YOLO depth estimation algorithm
'''      
class YOLO_ONNXRuntime_Depth(YOLO_ONNXRuntime):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (768, 768)
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
        depth = scale_mask(output, self.inputs_shape, self.image.shape)
        if self.draw_result:
            self.result = draw_result(depth)