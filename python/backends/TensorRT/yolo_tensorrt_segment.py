'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-01-12 10:49:13
Description: tensorrt inference class for YOLO segmentation algorithm
'''


from backends.utils import *
from backends.TensorRT.yolo_tensorrt import YOLO_TensorRT


'''
description: tensorrt inference class for the YOLO segmentation algorithm
'''             
class YOLO_TensorRT_Segment(YOLO_TensorRT):
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
        super().__init__(algo_type, device_type, model_type, model_path)
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output1_device = cupy.empty(self.outputs_shape[1], dtype=np.float32)
        self.output0_ptr = self.output0_device.data.ptr
        self.output1_ptr = self.output1_device.data.ptr
               
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''        
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape, use_cupy=True)
        input = cupy.transpose(input[:, :, ::-1], (2, 0, 1))
        input = input.astype(cupy.float32) / 255.0
        self.input_ptr = input.data.ptr
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.context.execute_v2(bindings=[self.input_ptr, self.output0_ptr, self.output1_ptr])
        self.output0_host = cupy.asnumpy(self.output0_device) 
        self.output1_host = cupy.asnumpy(self.output1_device) 

    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''            
    def post_process(self) -> None:
        if int(trt.__version__.split(".")[0]) < 10:
            output = np.squeeze(self.output1_host.reshape(self.outputs_shape[1]))
            proto = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
        else:
            output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
            proto = np.squeeze(self.output1_host.reshape(self.outputs_shape[1]))
        boxes = []
        scores = []
        
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
                          
        if len(boxes):   
            indices = nms(boxes, scores, self.iou_threshold) 
            boxes = boxes[indices]          
            masks_in = mask[indices]
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