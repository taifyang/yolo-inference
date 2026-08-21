'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-08-05 21:13:26
Description: pytorch inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO depth estimation algorithm
'''    
class YOLO_PyTorch_Depth(YOLO_PyTorch):
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
        depth = scale_mask(output.cpu().numpy(), self.inputs_shape, self.image.shape)
        if self.draw_result:
            self.result = draw_result(depth)