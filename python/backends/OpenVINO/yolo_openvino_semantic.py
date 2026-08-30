'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:26:38
Description: openvino inference class for YOLO semantic segmentation algorithm
'''


from backends.utils import *
from backends.OpenVINO.yolo_openvino import *


'''
description: openvino inference class for the YOLO semantic segmentation algorithm
'''       
class YOLO_OpenVINO_Semantic(YOLO_OpenVINO):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (1024, 1024)
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
        class_map = scale_masks(output, self.inputs_shape, self.image.shape)
        mask = class_map.astype(np.uint8)
        if self.draw_result:
            masks = []
            for cls_id in np.unique(mask):
                binary_mask = (mask == cls_id).astype(np.bool_)
                masks.append(binary_mask)
            self.result = draw_result(task_type='Semantic', image=self.image, masks=masks)