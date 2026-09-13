'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 00:13:44
Description: pytorch inference class for YOLO classifaction algorithm
'''


from backends.utils import *
from backends.yolo_classify import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO classifaction algorithm
'''     
class YOLO_PyTorch_Classify(YOLO_PyTorch, YOLO_Classify):
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
        YOLO_Classify.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        if self.algo_type in ['YOLOv5']:
            input = centercrop(self.image, self.inputs_shape)
            input = normalize(input, self.algo_type)
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            if self.image.shape[1] > self.image.shape[0]:
                input = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                input = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            input = centercrop(input, self.inputs_shape)
            input = normalize(input, self.algo_type)
            
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        self.inputs = np.expand_dims(input, axis=0).astype(dtype=np.float32)
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
        output = torch.squeeze(self.outputs).to(dtype=torch.float32)
        if self.algo_type in ['YOLOv5'] and self.render_result:
            class_id = torch.argmax(output).cpu().item()
            scores = (torch.exp(torch.max(output))/torch.sum(torch.exp(output))).cpu().item()
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'] and self.render_result:
            class_id = torch.argmax(output).cpu().item()
            scores = torch.max(output).cpu().item()
        if self.render_result:
            self.result = self.draw(class_id, scores)
    