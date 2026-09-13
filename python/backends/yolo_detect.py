'''
Author: taifyang
Date: 2026-09-10 23:38:07
LastEditTime: 2026-09-11 21:53:53
Description: class for YOLO detection algorithm
'''


import numpy as np
from backends.yolo import *


'''
description: class for YOLO detection algorithm
'''
class YOLO_Detect(YOLO):
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
        assert self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13', 'YOLO26'], 'algo type not supported!'

    def draw_result(self, preds):
        result = self.image.copy()   
        boxes = preds[..., :4] 
        scores = preds[..., 4]
        classes = preds[..., 5].astype(np.int32)
        
        for box, score, cls in zip(boxes, scores, classes):
            box = box.astype(np.int32)
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
        return result