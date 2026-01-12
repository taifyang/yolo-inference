'''
Author: taifyang
Date: 2026-01-09 23:31:45
LastEditTime: 2026-01-09 23:32:53
Description: pytorch inference class for YOLO obb algorithm
'''


from backends.utils import *
from backends.PyTorch.yolo_pytorch import YOLO_PyTorch


'''
description: pytorch inference class for the YOLO obb algorithm
'''      
class YOLO_PyTorch_OBB(YOLO_PyTorch):
    '''
    description:    construction method
    param {*} self  instance of class
    return {*}
    '''    
    def __init__(self, algo_type:str, device_type:str, model_type:str, model_path:str) -> None:
        super().__init__(algo_type, device_type, model_type, model_path)
        self.class_num = 15		            
        self.inputs_shape = (1024, 1024) 
        self.iou_threshold = 0.7

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
        boxes = []
        scores = []
        
        output = torch.squeeze(self.outputs[0]).to(torch.float32)
        xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
        box, cls, angle = output[xc].split((4, self.class_num, 1), 1)
        scores, j = cls.max(1, keepdim=True)
        boxes = torch.cat((box, scores, j, angle), 1)

        if len(boxes):   
            indices = nms_rotated(boxes, scores.squeeze(), self.iou_threshold) 
            boxes = boxes[indices]
            boxes = regularize_rboxes(boxes)
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape, xywh=True)
            boxes = reversed(boxes).cpu().numpy()     
            if self.draw_result:
                self.result = draw_result(self.image, boxes)