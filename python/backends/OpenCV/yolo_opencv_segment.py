'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-01-20 23:51:18
Description: opencv inference class for YOLO segmentation algorithm
'''


from backends.utils import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO segmentation algorithm
'''    
class YOLO_OpenCV_Segment(YOLO_OpenCV):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
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
        if self.algo_type in ['YOLOv5']:
            output = output[output[..., 4] > self.confidence_threshold]
            xc = np.max(output[...,5:(5+self.class_num)], axis=1) > self.score_threshold
            output[...,:4] = xywh2xyxy(output[...,:4])
            box = output[xc][:, :4]
            cls = output[xc][:, 5:(5+self.class_num)] 
            mask = output[xc][:, (5+self.class_num):]
            scores = np.max(cls, axis=1, keepdims=True) * output[..., 4:5]
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j.astype(np.float32)], axis=1)         
        elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            output = np.squeeze(output).astype(np.float32)
            cls_scores = output[..., 4:(4 + self.class_num)]
            xc = np.amax(cls_scores, axis=1) > self.score_threshold 
            output[..., :4] = xywh2xyxy(output[..., :4])
            box = output[xc][:, :4]
            cls = output[xc][:, 4:(4+self.class_num)]
            mask = output[xc][:, (4+self.class_num):]
            scores = np.max(cls, axis=1, keepdims=True) 
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j.astype(np.float32)], axis=1) 
        elif self.algo_type in ['YOLO26']:
            output = output[output[..., 4] > self.score_threshold]
            box = output[:, :4]
            scores = output[:, 4]
            boxes = output[:, :6]
            mask = output[:, 6:]    
                          
        if len(boxes):   
            if self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:  
                indices = cv2.dnn.NMSBoxes(boxes[..., :4], scores.squeeze(), self.score_threshold, self.iou_threshold)
                boxes = boxes[indices]          
                masks_in = mask[indices]
            proto = np.squeeze(self.outputs[1]).astype(dtype=np.float32)
            c, mh, mw = proto.shape 
            if self.algo_type in ['YOLOv5']:
                masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)  
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
                masks = (masks_in @ proto.reshape(c, -1)).reshape(-1, mh, mw)  
            elif self.algo_type in ['YOLO26']:
                masks = (mask @ proto.reshape(c, -1)).reshape(-1, mh, mw)   
            downsampled_bboxes = boxes.copy()
            downsampled_bboxes[:, 0] *= mw / self.inputs_shape[0]
            downsampled_bboxes[:, 2] *= mw / self.inputs_shape[0]
            downsampled_bboxes[:, 3] *= mh / self.inputs_shape[1]
            downsampled_bboxes[:, 1] *= mh / self.inputs_shape[1]       
            masks = crop_mask(masks, downsampled_bboxes)
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            resized_masks = []
            for mask in masks:
                mask = cv2.resize(mask, self.inputs_shape, cv2.INTER_LINEAR)
                mask = scale_mask(mask, self.inputs_shape, self.image.shape)
                resized_masks.append(mask)
            resized_masks = np.array(resized_masks)
            if self.algo_type in ['YOLOv5']:
                resized_masks = resized_masks > 0.5
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLO26']:
                resized_masks = resized_masks > 0       
            if self.draw_result:
                self.result = draw_result(self.image, boxes, resized_masks)
