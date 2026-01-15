'''
Author: taifyang
Date: 2026-01-12 10:57:45
LastEditTime: 2026-01-12 11:06:45
Description: tensorrt inference class for YOLO obb algorithm
'''


from backends.utils import *
from backends.TensorRT.yolo_tensorrt import *


'''
description: tensorrt inference class for the YOLO detection algorithm
'''   
class YOLO_TensorRT_OBB(YOLO_TensorRT):
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
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
        self.class_num = 15		            
        self.inputs_shape = (1024, 1024) 
        self.iou_threshold = 0.7
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output0_ptr = self.output0_device.data.ptr

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
        self.context.execute_v2(bindings=[self.input_ptr, self.output0_ptr])
        self.output0_host = cupy.asnumpy(self.output0_device) 

    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''  
    def post_process(self) -> None:
        boxes = []
        scores = []

        output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))  
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
  