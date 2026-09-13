'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-10 22:16:10
Description: pytorch inference class for YOLO detection algorithm
'''


from backends.utils import *
from backends.yolo_detect import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO detection algorithm
'''      
class YOLO_PyTorch_Detect(YOLO_PyTorch, YOLO_Detect):
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
        YOLO_PyTorch.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Detect.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13', 'YOLO26'], 'algo type not supported!'
        input = letterbox(self.image, self.inputs_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        self.inputs = np.expand_dims(input, axis=0) 
        self.inputs = torch.from_numpy(self.inputs)
        if self.device_type == 'GPU':
            self.inputs = self.inputs.cuda()
            if self.model_type == 'FP16':
                self.inputs = self.inputs.half()
        
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''     
    def post_process(self) -> None:  
        output = torch.squeeze(self.outputs[0]).to(torch.float32)     
        if self.algo_type in ['YOLOv3', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']:        
            xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
            if self.algo_type in ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLOv13']:
                output[..., :4] = xywh2xyxy(output[..., :4])  
            box, cls = output[xc].split((4, self.class_num), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores, j.float()), dim=1)
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:   
            output = output[output[..., 4] > self.confidence_threshold] 
            xc = output[..., 5:(5+self.class_num)].amax(1) > self.score_threshold
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box, obj, cls = output[xc].split((4, 1, self.class_num), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores*obj, j.float()), dim=1)
        elif self.algo_type in ['YOLO26']:
            boxes = output[output[..., 4] > self.score_threshold]
            box = boxes[:, :4]
            scores = boxes[:, 4]

        if len(boxes): 
            if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']:   
                indices = torchvision.ops.nms(box, scores, self.iou_threshold)
                boxes = boxes[indices]
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.render_result:
                self.result = self.draw_result(boxes.cpu().numpy())
