'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 21:51:10
Description: pytorch inference class for YOLO semantic segmentation algorithm
'''


from backends.utils import *
from backends.yolo_semantic import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO semantic segmentation algorithm
'''    
class YOLO_PyTorch_Semantic(YOLO_PyTorch, YOLO_Semantic):
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
        YOLO_PyTorch.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Semantic.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        self.inputs = np.expand_dims(input, axis=0)
        self.inputs = torch.from_numpy(self.inputs)
        if self.device_type == 'GPU':
            self.inputs = self.inputs.cuda()
            if self.model_type == 'FP16':
                self.inputs = self.inputs.half()
                
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output = self.outputs.to(torch.float32)
        class_map = scale_masks(output, self.inputs_shape, self.image.shape)
        mask = class_map[0].argmax(0).cpu().numpy().astype(np.uint8)
        if self.render_result:
            masks = []
            for cls_id in np.unique(mask):
                binary_mask = (mask == cls_id).astype(np.bool_)
                masks.append(binary_mask)
            self.result = self.draw_result(masks)
            