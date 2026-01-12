'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:28:19
Description: pytorch inference class for YOLO segmentation algorithm
'''


from backends.utils import *
from backends.PyTorch.yolo_pytorch import YOLO_PyTorch


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
            boxes = torch.cat((box, scores*obj, j.float()), dim=1)          
        elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            output = torch.squeeze(self.outputs[0]).to(torch.float32)
            xc = output[..., 4:(4+self.class_num)].amax(1) > self.score_threshold
            output[..., :4] = xywh2xyxy(output[..., :4])  
            box, cls, mask = output[xc].split((4, self.class_num, 32), 1)
            scores, j = cls.max(1, keepdim=True)
            boxes = torch.cat((box, scores, j.float()), dim=1)    
                          
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