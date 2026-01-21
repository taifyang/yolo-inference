'''
Author: taifyang
Date: 2026-01-05 11:09:10
LastEditTime: 2026-01-05 11:24:45
Description: pytorch inference class for YOLO pose algorithm
'''


from backends.utils import *
from backends.PyTorch.yolo_pytorch import *


'''
description: pytorch inference class for the YOLO pose algorithm
'''      
class YOLO_PyTorch_Pose(YOLO_PyTorch):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26'], 'algo type not supported!'
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
        if self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            scores = output[..., 4]
            xc = scores > self.score_threshold 
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box = output[xc][:, :4]
            scores = scores[xc].unsqueeze(1)
            cls = torch.zeros((len(box), 1)).to(output.device)
            kpts = output[xc][..., 5:]
            boxes = torch.cat((box, scores, cls, kpts), dim=1)
        elif self.algo_type in ['YOLO26']:
            boxes = output[output[..., 4] > self.score_threshold]
            box = boxes[:, :4]
            scores = boxes[..., 4:5]
        
        if len(boxes):   
            indices = torchvision.ops.nms(box, scores.squeeze(), self.iou_threshold)
            boxes = boxes[indices].cpu().numpy()
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.draw_result:
                self.result = draw_result(self.image, boxes, kpts=boxes[:, 6:])
