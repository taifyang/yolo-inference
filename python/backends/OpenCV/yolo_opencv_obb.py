'''
Author: taifyang
Date: 2026-01-12 10:46:21
LastEditTime: 2026-01-12 10:46:37
Description: opencv inference class for YOLO pose algorithm
'''


from backends.utils import *
from backends.OpenCV.yolo_opencv import YOLO_OpenCV


'''
description: onnxruntime inference class for the YOLO obb algorithm
'''
class YOLO_OpenCV_OBB(YOLO_OpenCV):
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
        self.inputs = cv2.dnn.blobFromImage(input, 1/255., size=self.inputs_shape, swapRB=True, crop=False)
        self.net.setInput(self.inputs)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        boxes = []
        scores = []

        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']: 
            output = np.squeeze(self.outputs[0]).astype(np.float32)
            cls_scores = output[..., 4:(4 + self.class_num)]
            xc = np.amax(cls_scores, axis=1) > self.score_threshold 
            box = output[xc][:, :4]
            cls = output[xc][:, 4:(4+self.class_num)]
            angle = output[xc][:, -1:]
            scores = np.max(cls, axis=1, keepdims=True) 
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j, angle], axis=1)
             
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
                