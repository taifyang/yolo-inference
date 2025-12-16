'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-12-16 20:20:53
FilePath: \python\backends\PyTorch_\yolo_pytorch.py
Description: pytorch inference class for YOLO algorithm
'''

import torch
import torchvision
import torch.nn.functional as F
from backends.yolo import *
from backends.utils import *


'''
description: pytorch inference class for YOLO algorithm
'''
class YOLO_PyTorch(YOLO):
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
        self.net = torch.jit.load(model_path)
        assert self.net is not None, 'load model failed!'
        self.algo_type = algo_type
        self.device_type = device_type
        self.model_type = model_type
        if self.device_type == 'GPU':
            self.net = self.net.cuda()
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.outputs = self.net(self.inputs)

'''
description: pytorch inference class for the YOLO classfiy algorithm
'''     
class YOLO_PyTorch_Classify(YOLO_PyTorch):
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
        self.inputs = np.expand_dims(input, axis=0).astype(dtype=np.float32)
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
        output = torch.squeeze(self.outputs).to(dtype=torch.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', torch.argmax(output).cpu().item(), \
                  ' scores:', (torch.exp(torch.max(output))/torch.sum(torch.exp(output))).cpu().item())
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
            print('class:', torch.argmax(output).cpu().item(), ' scores:', torch.max(output).cpu().item())
    

'''
description: pytorch inference class for the YOLO detection algorithm
'''      
class YOLO_PyTorch_Detect(YOLO_PyTorch):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13'], 'algo type not supported!'
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
        boxes = []
        scores = []

        if self.algo_type in ['YOLOv3', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']: 
            output = torch.squeeze(self.outputs[0]).to(torch.float32)
            xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
            if self.algo_type in ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12', 'YOLOv13']:
                output[..., :4] = xywh2xyxy(output[..., :4])  
            box, cls = output[xc].split((4, self.class_num), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores, j.float()), 1)
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:   
            output = torch.squeeze(self.outputs[0]).to(torch.float32)
            output = output[output[..., 4] > self.confidence_threshold] 
            xc = output[..., 5:(5+self.class_num)].amax(1) > self.score_threshold
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box, obj, cls = output[xc].split((4, 1, self.class_num), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores*obj, j.float()), 1)
             
        if len(boxes):   
            indices = torchvision.ops.nms(box, scores.squeeze(), self.iou_threshold)
            boxes = boxes[indices].cpu().numpy()
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            if self.draw_result:
                self.result = draw_result(self.image, boxes)


'''
description: pytorch inference class for the YOLO segmentation algorithm
'''    
class YOLO_PyTorch_Segment(YOLO_PyTorch):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9','YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
        boxes = []
        scores = []

        if self.algo_type in ['YOLOv5']:
            output = torch.squeeze(self.outputs[0]).to(torch.float32)
            output = output[output[..., 4] > self.confidence_threshold] 
            xc = output[..., 5:(5+self.class_num)].amax(1) > self.score_threshold
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box, obj, cls, mask = output[xc].split((4, 1, self.class_num, 32), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores*obj, j.float()), 1)          
        elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            output = torch.squeeze(self.outputs[0]).to(torch.float32)
            xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box, cls, mask = output[xc].split((4, self.class_num, 32), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores, j.float()), 1)    
                          
        if len(boxes):   
            indices = torchvision.ops.nms(box, scores.squeeze(), self.iou_threshold)
            boxes = boxes[indices]       
            masks_in = mask[indices]
            proto = torch.squeeze(self.outputs[1]).to(dtype=torch.float32)
            c, mh, mw = proto.shape 
            if self.algo_type in ['YOLOv5']:
                masks = (1/ (1 + torch.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)  
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
                masks = (masks_in @ proto.reshape(c, -1)).reshape(-1, mh, mw)    
            downsampled_bboxes = boxes.clone()
            downsampled_bboxes[:, 0] *= mw / self.inputs_shape[0]
            downsampled_bboxes[:, 2] *= mw / self.inputs_shape[0]
            downsampled_bboxes[:, 1] *= mh / self.inputs_shape[1] 
            downsampled_bboxes[:, 3] *= mh / self.inputs_shape[1]
          
            masks = crop_mask(masks, downsampled_bboxes)
            boxes = scale_boxes(boxes, self.inputs_shape, self.image.shape)
            masks = F.interpolate(masks[None], self.inputs_shape, mode="bilinear", align_corners=False)[0]
            resized_masks = []
            for mask in masks:
                resized_mask = scale_mask(mask.cpu().numpy(), self.inputs_shape, self.image.shape)
                resized_masks.append(resized_mask)
            resized_masks = np.array(resized_masks)
            if self.algo_type in ['YOLOv5']:
                resized_masks = resized_masks > 0.5
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
                resized_masks = resized_masks > 0    
            if self.draw_result:
                self.result = draw_result(self.image, boxes.cpu().numpy(), resized_masks.astype(np.bool_))