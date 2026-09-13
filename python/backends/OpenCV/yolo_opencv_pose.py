'''
Author: taifyang
Date: 2026-01-03 23:46:27
LastEditTime: 2026-09-13 21:36:04
Description: opencv inference class for YOLO pose algorithm
'''


from backends.utils import *
from backends.yolo_pose import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO pose algorithm
'''      
class YOLO_OpenCV_Pose(YOLO_OpenCV, YOLO_Pose):
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
        YOLO_OpenCV.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Pose.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
        input = letterbox(self.image, self.inputs_shape)
        self.inputs = cv2.dnn.blobFromImage(input, 1/255., size=self.inputs_shape, swapRB=True, crop=False)
        self.net.setInput(self.inputs)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''     
    def post_process(self) -> None:
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32) 
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            scores = output[..., 4]
            xc = scores > self.score_threshold 
            output[..., :4] = xywh2xyxy(output[..., :4])
            box = output[xc][:, :4]
            scores =  np.expand_dims(scores[xc], axis=1)
            cls = np.zeros((len(box), 1))
            kpts = output[xc][..., 5:]
        elif self.algo_type in ['YOLO26']:
            output = output[output[..., 4] > self.score_threshold]
            box = output[:, :4]
            scores = output[..., 4:5]
            cls = output[..., 5:6]
            kpts = output[..., 6:]
        boxes = np.concatenate([box, scores, cls, kpts], axis=1)
        
        if len(boxes):
            if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:    
                indices = nms(boxes, scores, self.iou_threshold) 
                boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.render_result:
                self.result = self.draw_result(boxes) 
