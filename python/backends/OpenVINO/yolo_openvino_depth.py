'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-13 21:32:58
Description: openvino inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.yolo_depth import *
from backends.OpenVINO.yolo_openvino import *


'''
description: openvino inference class for the YOLO depth estimation algorithm
'''       
class YOLO_OpenVINO_Depth(YOLO_OpenVINO, YOLO_Depth):
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
        YOLO_OpenVINO.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Depth.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (768, 768)
        input = letterbox(self.image, self.inputs_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        self.inputs = np.expand_dims(input, axis=0)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output0 = self.outputs[self.compiled_model.output(0)]
        output = np.squeeze(output0).astype(dtype=np.float32)
        depth = scale_masks(output, self.inputs_shape, self.image.shape)
        if self.render_result:
            self.result = self.draw_result(depth)