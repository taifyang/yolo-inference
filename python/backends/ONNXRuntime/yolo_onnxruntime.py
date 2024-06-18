'''
Author: taifyang 58515915+taifyang@users.noreply.github.com
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-06-18 21:56:16
Description: yolo算法onnxruntime推理框架实现
'''

import onnxruntime
from backends.yolo import *
from backends.utils import *


'''
description: yolo算法onnxruntime推理框架实现类
'''
class YOLO_ONNXRuntime(YOLO):
    '''
    description:            构造方法
    param {*} self
    param {str} algo_type   算法类型
    param {str} device_type 设备类型
    param {str} model_type  模型精度
    param {str} model_path  模型路径
    return {*}
    '''    
    def __init__(self, algo_type:str, device_type:str, model_type:str, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), "model not exists!"
        if device_type == 'CPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        elif device_type == 'GPU':
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.algo_type = algo_type
        self.model_type = model_type
         
        self.input_name = []
        for node in self.onnx_session.get_inputs(): 
            self.input_name.append(node.name)
        self.output_name = []
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)
        self.input = {}
    
    '''
    description:    模型推理
    param {*} self
    return {*}
    '''    
    def process(self) -> None:
        self.output = self.onnx_session.run(None, self.input)


'''
description: yolo分类算法onnxruntime推理框架实现类
'''
class YOLO_ONNXRuntime_Classify(YOLO_ONNXRuntime):   
    '''
    description:    模型前处理
    param {*} self
    return {*}
    '''            
    def pre_process(self) -> None:
        if self.algo_type == 'YOLOv5':
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
            input = input - np.array([0.406, 0.456, 0.485])
            input = input / np.array([0.225, 0.224, 0.229])
        if self.algo_type == 'YOLOv8':
            self.input_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                self.image = cv2.resize(self.image, (self.input_shape[0]*self.image.shape[1]//self.image.shape[0], self.input_shape[0]))
            else:
                self.image = cv2.resize(self.image, (self.input_shape[1], self.input_shape[1]*self.image.shape[0]//self.image.shape[1]))
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        if self.model_type == 'FP32' or self.model_type == 'INT8':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == 'FP16':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
    
    '''
    description:    模型后处理
    param {*} self
    return {*}
    '''           
    def post_process(self) -> None:
        output = np.squeeze(self.output).astype(dtype=np.float32)
        if self.algo_type == 'YOLOv5':
            print("class:", np.argmax(output), " scores:", np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type == 'YOLOv8':
            print("class:", np.argmax(output), " scores:", np.max(output))


'''
description: yolo检测算法onnxruntime推理框架实现类
'''
class YOLO_ONNXRuntime_Detect(YOLO_ONNXRuntime):
    '''
    description:    模型前处理
    param {*} self
    return {*}
    '''    
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        if self.model_type == 'FP32' or self.model_type == 'INT8':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == 'FP16':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
    
    '''
    description:    模型后处理
    param {*} self
    return {*}
    '''         
    def post_process(self) -> None:
        output = np.squeeze(self.output[0]).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        if self.algo_type == 'YOLOv5':
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:85]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output[i][4]
                cls_score = classes_scores[i][class_id]
                output[i][4] = obj_score * cls_score
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i][:6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])   
                    output[i][5:] *= obj_score
        if self.algo_type == 'YOLOv8': 
            for i in range(output.shape[0]):
                classes_scores = output[..., 4:]     
                class_id = np.argmax(classes_scores[i])
                output[i][4] = classes_scores[i][class_id]
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i, :6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])                  
        if len(boxes):   
            boxes = np.array(boxes)
            boxes = xywh2xyxy(boxes)
            scores = np.array(scores)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
            boxes = boxes[indices]
            self.result = draw(self.image, boxes)
            

'''
description: yolo分割算法onnxruntime推理框架实现类
'''      
class YOLO_ONNXRuntime_Segment(YOLO_ONNXRuntime):
    '''
    description:    模型前处理
    param {*} self
    return {*}
    '''    
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        if self.model_type == 'FP32' or self.model_type == 'INT8':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == 'FP16':
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
    
    '''
    description:    模型后处理
    param {*} self
    return {*}
    '''           
    def post_process(self) -> None:
        output = np.squeeze(self.output[0]).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type == 'YOLOv5':
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:85]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output[i][4]
                cls_score = classes_scores[i][class_id]
                output[i][4] = obj_score * cls_score
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i][:6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])   
                    output[i][5:] *= obj_score
                    preds.append(output[i])
        if self.algo_type == 'YOLOv8': 
            for i in range(output.shape[0]):
                classes_scores = output[..., 4:84]     
                class_id = np.argmax(classes_scores[i])
                output[i][4] = classes_scores[i][class_id]
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i, :6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])    
                    preds.append(output[i])           
        if len(boxes):   
            boxes = np.array(boxes)
            boxes = xywh2xyxy(boxes)
            scores = np.array(scores)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
            boxes = boxes[indices]
            
            masks_in = np.array(preds)[indices][..., -32:]
            proto= np.squeeze(self.output[1]).astype(dtype=np.float32)
            c, mh, mw = proto.shape 
            masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)
            
            downsampled_bboxes = boxes.copy()
            downsampled_bboxes[:, 0] *= mw / self.input_shape[0]
            downsampled_bboxes[:, 2] *= mw / self.input_shape[0]
            downsampled_bboxes[:, 3] *= mh / self.input_shape[1]
            downsampled_bboxes[:, 1] *= mh / self.input_shape[1]
        
            masks = crop_mask(masks, downsampled_bboxes)
            self.result = draw(self.image, boxes, masks)