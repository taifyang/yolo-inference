'''
Author: taifyang
Date: 2026-09-10 21:56:34
LastEditTime: 2026-09-11 20:51:20
Description: class for YOLO classifaction algorithm
'''


from backends.yolo import *


'''
description: class for YOLO classifaction algorithm
'''
class YOLO_Classify(YOLO):
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
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            self.inputs_shape = (224, 224)

    def draw_result(self, class_id, scores):
        result = self.image.copy()
        label = f"class{class_id}:{scores:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(result, label, (0, label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        return result