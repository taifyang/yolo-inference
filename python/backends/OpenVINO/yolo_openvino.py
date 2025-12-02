'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-10-13 21:24:25
FilePath: \python\backends\OpenVINO\yolo_openvino.py
Description: openvino inference class for YOLO algorithm
'''

import openvino as ov
from backends.yolo import *
from backends.utils import *


'''
description: openvino inference class for YOLO algorithm
'''
class YOLO_OpenVINO(YOLO):
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
        try:
            from openvino.runtime import Core
            core = Core()
        except:
            core = ov.Core()
        model  = core.read_model(model_path)
        self.algo_type = algo_type
        self.compiled_model = core.compile_model(model, device_name='GPU' if device_type=='GPU' else 'CPU')
        assert self.compiled_model, 'compile model failed!'
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.compiled_model({0: self.inputs})


'''
description: openvino inference class for the YOLO classfiy algorithm
'''
class YOLO_OpenVINO_Classify(YOLO_OpenVINO):
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
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                self.image = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                self.image = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = crop_image / 255.0
            
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        self.inputs = np.expand_dims(input, axis=0)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output = self.outputs[self.compiled_model.output(0)]
        output = np.squeeze(output).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.max(output))
       
 
'''
description: openvino inference class for the YOLO detection algorithm
'''   
class YOLO_OpenVINO_Detect(YOLO_OpenVINO):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13'], 'algo type not supported!'
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
        boxes = []
        scores = []
        class_ids = []
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)

        if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']: 
            classes_scores = output[..., 4:(4 + self.class_num)]  
            class_ids = np.argmax(classes_scores, axis=-1)  
            scores_all = np.max(classes_scores, axis=-1)        
            mask = scores_all > self.score_threshold  
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()   
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5 + self.class_num)]
            class_ids = np.argmax(classes_scores, axis=-1)
            class_scores = np.max(classes_scores, axis=-1)
            scores_all = class_scores * output[..., 4]        
            mask = scores_all > self.score_threshold
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze() 
             
        if len(boxes):   
            boxes = np.array(boxes)
            scores = np.array(scores)
            if self.algo_type in ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLOv13']:
                boxes = xywh2xyxy(boxes)
            elif self.algo_type in ['YOLOv4']:
                boxes[..., [0, 2]] *= self.inputs_shape[0]
                boxes[..., [1, 3]] *= self.inputs_shape[1]
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
            boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.draw_result:
                self.result = draw_result(self.image, boxes)
        

'''
description: openvino inference class for the YOLO segmentation algorithm
'''       
class YOLO_OpenVINO_Segment(YOLO_OpenVINO):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
        output0 = self.outputs[self.compiled_model.output(0)]
        output = np.squeeze(output0).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []

        if self.algo_type in ['YOLOv5']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5 + self.class_num)]
            class_ids = np.argmax(classes_scores, axis=-1)
            class_scores = np.max(classes_scores, axis=-1)
            scores_all = class_scores * output[..., 4]        
            mask = scores_all > self.score_threshold
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()    
            preds = output[mask]           
        elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            classes_scores = output[..., 4:(4 + self.class_num)]  
            class_ids = np.argmax(classes_scores, axis=-1)  
            scores_all = np.max(classes_scores, axis=-1)        
            mask = scores_all > self.score_threshold  
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()     
            preds = output[mask]    
                    
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
            if self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
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
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
                resized_masks = resized_masks > 0       
            if self.draw_result:
                self.result = draw_result(self.image, boxes, resized_masks)