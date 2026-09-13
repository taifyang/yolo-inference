'''
Author: taifyang
Date: 2026-09-10 23:38:07
LastEditTime: 2026-09-11 21:54:10
Description: class for YOLO obb algorithm
'''


import numpy as np
from backends.yolo import *
from backends.utils import *


'''
description: class for YOLO obb algorithm
'''
class YOLO_OBB(YOLO):
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
        YOLO.__init__(self)
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
        self.class_num = 15		            
        self.inputs_shape = (1024, 1024) 
        self.iou_threshold = 0.7

    def draw_result(self, preds):
        result = self.image.copy()   
        boxes = preds[..., :4] 
        scores = preds[..., 4]
        classes = preds[..., 5].astype(np.int32)

        boxes = np.concatenate((boxes, preds[..., -1:]), axis=1)   
        for box, score, cls in zip(boxes, scores, classes):
            box = xywhr2xyxyxyxy(box).astype(np.int32)
            cv2.polylines(result, [np.asarray(box)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)     
        return result