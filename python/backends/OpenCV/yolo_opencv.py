'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-09-04 11:20:01
FilePath: \python\backends\OpenCV\yolo_opencv.py
Description: opencv inference class for YOLO algorithm
'''

import cv2
from backends.yolo import *
from backends.utils import *


'''
description: opencv inference class for YOLO algorithm
'''
class YOLO_OpenCV(YOLO):
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
        super().__init__()
        assert os.path.exists(model_path), 'model not exists!'
        assert device_type in ['CPU', 'GPU'], 'unsupported device type!'
        assert model_type in ['FP32', 'FP16'], 'unsupported model type!'
        self.net = cv2.dnn.readNet(model_path)
        self.algo_type = algo_type
        if device_type == 'CPU':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if device_type == 'GPU':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            if model_type == 'FP32':
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            if model_type == 'FP16':
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())


'''
description: opencv inference class for the YOLO classfiy algorithm
'''     
class YOLO_OpenCV_Classify(YOLO_OpenCV):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
        if self.algo_type in ['YOLOv5']:
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.inputs_shape)
            input = input / 255.0
            input = input - np.array([0.406, 0.456, 0.485])
            input = input / np.array([0.225, 0.224, 0.229])
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                self.image = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                self.image = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.inputs_shape)
            input = input / 255.0
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        self.inputs = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        self.net.setInput(self.inputs)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''    
    def post_process(self) -> None:
        output = np.squeeze(self.outputs).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.max(output))
    

'''
description: opencv inference class for the YOLO detection algorithm
'''      
class YOLO_OpenCV_Detect(YOLO_OpenCV):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13'], 'algo type not supported!'
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
        boxes = []
        scores = []
        class_ids = []

        if self.algo_type in ['YOLOv5', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5+self.class_num)]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id] * output[i][4]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
        if self.algo_type in ['YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']: 
            classes_scores = output[..., 4:(4+self.class_num)]          
            for i in range(output.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id)    
             
        if len(boxes):   
            boxes = np.array(boxes)
            scores = np.array(scores)
            if self.algo_type in ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLOv13']:
                boxes = xywh2xyxy(boxes)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
            boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.draw_result:
                self.result = draw_result(self.image, boxes)


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
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type in ['YOLOv5']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5+self.class_num)]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id] * output[i][4]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output[i])  
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']: 
            classes_scores = output[..., 4:(4+self.class_num)]          
            for i in range(output.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output[i])    
                          
        if len(boxes):   
            boxes = np.array(boxes)
            boxes = xywh2xyxy(boxes)
            scores = np.array(scores)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold)
            boxes = boxes[indices]          
            masks_in = np.array(preds)[indices][..., -32:]
            proto = np.squeeze(self.outputs[1]).astype(dtype=np.float32)
            c, mh, mw = proto.shape 
            if self.algo_type in ['YOLOv5']:
                masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)  
            if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
                masks = (masks_in @ proto.reshape(c, -1)).reshape(-1, mh, mw)    
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
            if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
                resized_masks = resized_masks > 0       
            if self.draw_result:
                self.result = draw_result(self.image, boxes, resized_masks)