'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-13 21:33:06
Description: pytorch inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.yolo_depth import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO depth estimation algorithm
'''    
class YOLO_PyTorch_Depth(YOLO_PyTorch, YOLO_Depth):
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
        output = torch.squeeze(self.outputs[0]).to(torch.float32)
        depth = scale_masks(output.cpu().numpy(), self.inputs_shape, self.image.shape)
        if self.render_result:
            self.result = self.draw_result(depth)