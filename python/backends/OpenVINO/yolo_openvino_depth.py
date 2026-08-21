'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-08-04 00:11:28
Description: openvino inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.OpenVINO.yolo_openvino import *


'''
description: openvino inference class for the YOLO depth estimation algorithm
'''       
class YOLO_OpenVINO_Depth(YOLO_OpenVINO):
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
        depth = scale_mask(output, self.inputs_shape, self.image.shape)
        if self.draw_result:
            self.result = draw_result(depth)