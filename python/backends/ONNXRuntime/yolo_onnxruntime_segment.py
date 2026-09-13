'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 22:40:52
Description: onnxruntime inference class for YOLO segmentation algorithm
'''


from backends.utils import *
from backends.yolo_segment import *
from backends.ONNXRuntime.yolo_onnxruntime import *
            

'''
description: onnxruntime inference class for the YOLO segmentation algorithm
'''      
class YOLO_ONNXRuntime_Segment(YOLO_ONNXRuntime, YOLO_Segment):
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
        YOLO_Segment.__init__(self, algo_type, device_type, model_type, model_path)

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
                indices = nms(boxes, scores, self.iou_threshold) 
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
                mask = scale_masks(mask, self.inputs_shape, self.image.shape)
                resized_masks.append(mask)
            resized_masks = np.array(resized_masks)
            if self.algo_type in ['YOLOv5']:
                resized_masks = resized_masks > 0.5
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLO26']:
                resized_masks = resized_masks > 0       
            if self.render_result:
                self.result = self.draw_result(boxes, resized_masks)