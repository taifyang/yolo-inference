'''
Author: taifyang
Date: 2026-01-03 23:46:27
LastEditTime: 2026-01-03 23:49:18
Description: opencv inference class for YOLO pose algorithm
'''


from backends.utils import *
from backends.OpenCV.yolo_opencv import YOLO_OpenCV


'''
description: opencv inference class for the YOLO pose algorithm
'''      
class YOLO_OpenCV_Pose(YOLO_OpenCV):
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
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)    
        scores = output[..., 4]
        xc = scores > self.score_threshold 
        output[..., :4] = xywh2xyxy(output[..., :4])
        box = output[xc][:, :4]
        scores =  np.expand_dims(scores[xc], axis=1)
        cls = np.zeros((len(box), 1))
        kpts = output[xc][..., 5:]
        boxes = np.concatenate([box, scores, cls, kpts], axis=1)
        
        if len(boxes):   
            indices = nms(boxes, scores, self.iou_threshold) 
            boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.draw_result:
                self.result = draw_result(self.image, boxes, kpts=boxes[:, 6:])
