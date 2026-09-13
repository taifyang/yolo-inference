'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-10 23:41:17
Description: onnxruntime inference class for YOLO detection algorithm
'''


from backends.utils import *
from backends.yolo_detect import *
from backends.ONNXRuntime.yolo_onnxruntime import *


'''
description: onnxruntime inference class for the YOLO detection algorithm
'''
class YOLO_ONNXRuntime_Detect(YOLO_ONNXRuntime, YOLO_Detect):
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
        YOLO_ONNXRuntime.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Detect.__init__(self, algo_type, device_type, model_type, model_path)
        
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        if self.model_type == 'FP32' or self.model_type == 'INT8':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == 'FP16':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        self.inputs[self.inputs_name[0]] = input
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''         
    def post_process(self) -> None:
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)     
        if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']:  
            output = np.squeeze(output).astype(np.float32)
            cls_scores = output[..., 4:(4 + self.class_num)]
            xc = np.amax(cls_scores, axis=1) > self.score_threshold 
            if self.algo_type in ['YOLOv4']:
                box = output[xc][:, :4] 
                box[..., [0, 2]] *= self.inputs_shape[0]
                box[..., [1, 3]] *= self.inputs_shape[1]
            else:
                output[..., :4] = xywh2xyxy(output[..., :4])
                box = output[xc][:, :4]
            cls = output[xc][:, 4:(4+self.class_num)]
            scores = np.max(cls, axis=1, keepdims=True) 
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j.astype(np.float32)], axis=1)
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            xc = np.max(output[...,5:(5+self.class_num)], axis=1) > self.score_threshold
            output[...,:4] = xywh2xyxy(output[...,:4])
            box = output[xc][:, :4]
            cls = output[xc][:, 5:(5+self.class_num)] 
            scores = np.max(cls, axis=1, keepdims=True) * output[..., 4:5]
            j = np.argmax(cls, axis=1, keepdims=True) 
            boxes = np.concatenate([box, scores, j.astype(np.float32)], axis=1) 	
        elif self.algo_type in ['YOLO26']:
            boxes = output[output[..., 4] > self.score_threshold]
            box = boxes[:, :4]
            scores = boxes[:, 4]
             
        if len(boxes):
            if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']:   
                indices = nms(boxes, scores, self.iou_threshold) 
                boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.render_result:
                self.result = self.draw_result(boxes)
            