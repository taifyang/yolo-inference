'''
Author: taifyang
Date: 2026-01-12 10:52:15
LastEditTime: 2026-01-12 10:52:25
Description: openvino inference class for YOLO obb algorithm
'''


from backends.utils import *
from backends.OpenVINO.yolo_openvino import *

 
'''
description: openvino inference class for the YOLO obb algorithm
'''   
class YOLO_OpenVINO_OBB(YOLO_OpenVINO):
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
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
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
        output = np.squeeze(self.outputs[0]).astype(np.float32)
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']: 
            cls_scores = output[..., 4:(4 + self.class_num)]
            xc = np.amax(cls_scores, axis=1) > self.score_threshold 
            box = output[xc][:, :4]
            cls = output[xc][:, 4:(4+self.class_num)]
            angle = output[xc][:, -1:]
            scores = np.max(cls, axis=1, keepdims=True) 
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j, angle], axis=1)
        elif self.algo_type in ['YOLO26']:
            boxes = output[output[..., 4] > self.score_threshold]
            scores = boxes[..., 4:5]
                
        if len(boxes):   
            boxes = np.array(boxes)
            scores = np.array(scores).squeeze()
            indices = nms_rotated(boxes, scores, self.iou_threshold)   
            boxes = boxes[indices]
            boxes = regularize_rboxes(boxes)
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape, xywh=True)
            boxes = np.array(list(reversed(boxes)))
            if self.draw_result:
                self.result = draw_result(self.image, boxes)   
