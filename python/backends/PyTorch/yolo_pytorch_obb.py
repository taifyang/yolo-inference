'''
Author: taifyang
Date: 2026-01-09 23:31:45
LastEditTime: 2026-01-09 23:32:53
Description: pytorch inference class for YOLO obb algorithm
'''


from backends.utils import *
from backends.yolo_obb import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO obb algorithm
'''      
class YOLO_PyTorch_OBB(YOLO_PyTorch, YOLO_OBB):
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
        YOLO_OBB.__init__(self, algo_type, device_type, model_type, model_path)

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
        output = torch.squeeze(self.outputs[0]).to(torch.float32)
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']: 
            xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
            box, cls, angle = output[xc].split((4, self.class_num, 1), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores, j, angle), 1)
        elif self.algo_type in ['YOLO26']:
            boxes = output[output[..., 4] > self.score_threshold]
            scores = boxes[..., 4:5]

        if len(boxes):   
            indices = nms_rotated(boxes, scores.squeeze(), self.iou_threshold) 
            boxes = boxes[indices]
            boxes = regularize_rboxes(boxes)
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape, xywh=True)
            boxes = reversed(boxes).cpu().numpy()     
            if self.render_result:
                self.result = self.draw_result(boxes)  