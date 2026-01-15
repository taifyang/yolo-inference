'''
Author: taifyang
Date: 2026-01-03 23:28:03
LastEditTime: 2026-01-15 23:18:04
Description: onnxruntime inference class for YOLO pose algorithm
'''


from backends.utils import *
from backends.ONNXRuntime.yolo_onnxruntime import *


'''
description: onnxruntime inference class for the YOLO pose algorithm
'''
class YOLO_ONNXRuntime_Pose(YOLO_ONNXRuntime):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
