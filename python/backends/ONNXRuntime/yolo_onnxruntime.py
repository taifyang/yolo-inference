'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-10-15 21:02:55
FilePath: \python\backends\ONNXRuntime\yolo_onnxruntime.py
Description: onnxruntime inference class for YOLO algorithm
'''


import onnxruntime
from backends.yolo import *
from backends.utils import *


'''
description: onnxruntime inference class for YOLO algorithm
'''
class YOLO_ONNXRuntime(YOLO):
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
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, os.cpu_count() // 2)
        if device_type == 'CPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        elif device_type == 'GPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CUDAExecutionProvider'])
        assert self.onnx_session is not None, 'onnx_session is None!'
        self.algo_type = algo_type
        self.model_type = model_type    
        self.inputs_name = []
        for node in self.onnx_session.get_inputs(): 
            self.inputs_name.append(node.name)
        self.inputs = {}
    
    '''
    description:    model infenence
    param {*} self  instance of class
    return {*}
    '''    
    def process(self) -> None:
        self.outputs = self.onnx_session.run(None, self.inputs)

'''
description: onnxruntime inference class for the YOLO classfiy algorithm
'''
class YOLO_ONNXRuntime_Classify(YOLO_ONNXRuntime):   
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
            input = cv2.resize(crop_image, self.inputs_shape)
            input = input / 255.0
            
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
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
        output = np.squeeze(self.outputs).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print("class:", np.argmax(output), " scores:", np.exp(np.max(output))/np.sum(np.exp(output)))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
            print("class:", np.argmax(output), " scores:", np.max(output))


'''
description: onnxruntime inference class for the YOLO detection algorithm
'''
class YOLO_ONNXRuntime_Detect(YOLO_ONNXRuntime):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13'], 'algo type not supported!'
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
        boxes = []
        scores = []
        class_ids = []
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)
        
        if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']: 
            classes_scores = output[..., 4:(4+self.class_num)]          
            for i in range(output.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5+self.class_num)]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                scores.append(score)
                class_ids.append(class_id)     	
             
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
description: onnxruntime inference class for the YOLO segmentation algorithm
'''      
class YOLO_ONNXRuntime_Segment(YOLO_ONNXRuntime):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
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
        boxes = []
        scores = []
        class_ids = []
        preds = []
        
        if self.algo_type in ['YOLOv5']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5+self.class_num)]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id] * output[i][4]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output[i])  
        if self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            classes_scores = output[..., 4:(4+self.class_num)]          
            for i in range(output.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output[i])  
                          
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
