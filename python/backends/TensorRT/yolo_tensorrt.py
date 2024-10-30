'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang
LastEditTime: 2024-10-30 22:54:14
FilePath: \python\backends\TensorRT\yolo_tensorrt.py
Description: tensorrt inference class for YOLO algorithm
'''

import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda  
from backends.yolo import *
from backends.utils import *


'''
description: tensorrt inference class for YOLO algorithm
'''
class YOLO_TensorRT(YOLO):
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
        assert device_type in ['GPU'], 'unsupported device type!'
        self.algo_type = algo_type
        self.model_type = model_type
        logger = trt.Logger(trt.Logger.ERROR)
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.stream = cuda.Stream()
   

'''
description: tensorrt inference class for the YOLO classfiy algorithm
'''        
class YOLO_TensorRT_Classify(YOLO_TensorRT):
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
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11'], 'algo type not supported!'
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
    
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''       
    def pre_process(self) -> None:
        if self.algo_type in ['YOLOv5']:
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
            input = input - np.array([0.406, 0.456, 0.485])
            input = input / np.array([0.225, 0.224, 0.229])
        if self.algo_type in ['YOLOv8', 'YOLOv11']:
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
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()  
            self.output = self.output_host.reshape(context.get_binding_shape(1)) 
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''          
    def post_process(self) -> None:
        output = np.squeeze(self.output).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type in ['YOLOv8', 'YOLOv11'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.max(output))
    
 
'''
description: tensorrt inference class for the YOLO detection algorithm
'''   
class YOLO_TensorRT_Detect(YOLO_TensorRT):
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
        assert self.algo_type in ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11'], 'algo type not supported!'
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
    
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''       
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()  
            self.output = self.output_host.reshape(context.get_binding_shape(1)) 
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''       
    def post_process(self) -> None:
        output = np.squeeze(self.output[0]).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        if self.algo_type in ['YOLOv5', 'YOLOv6', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5+self.class_num)]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id] * output[i][4]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
        if self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11']: 
            classes_scores = output[..., 4:(4+self.class_num)]          
            for i in range(output.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id)    
                               
        if len(boxes):   
            boxes = np.array(boxes)
            scores = np.array(scores)
            if self.algo_type in ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv11']:
                boxes = xywh2xyxy(boxes)
                indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
                boxes = boxes[indices]
            if self.draw_result:
                self.result = draw_result(self.image, boxes, input_shape=self.input_shape)
        

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
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11'], 'algo type not supported!'
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output0_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.output1_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output0_device = cuda.mem_alloc(self.output0_host.nbytes)
        self.output1_device = cuda.mem_alloc(self.output1_host.nbytes)
    
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''        
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB and HWC2CHW
        input = input / 255.0
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
    
    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output0_device), int(self.output1_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output0_host, self.output0_device, self.stream)
            cuda.memcpy_dtoh_async(self.output1_host, self.output1_device, self.stream)
            self.stream.synchronize()  
            self.output0 = self.output0_host.reshape(context.get_binding_shape(1)) 
            self.output1 = self.output1_host.reshape(context.get_binding_shape(2)) 
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''            
    def post_process(self) -> None:
        output1 = np.squeeze(self.output1).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type in ['YOLOv5']:
            output1 = output1[output1[..., 4] > self.confidence_threshold]
            classes_scores = output1[..., 5:(5+self.class_num)]     
            for i in range(output1.shape[0]):
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id] * output1[i][4]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output1[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output1[i])                            
        if self.algo_type in ['YOLOv8', 'YOLOv11']: 
            classes_scores = output1[..., 4:(4+self.class_num)]   
            for i in range(output1.shape[0]):              
                class_id = np.argmax(classes_scores[i])
                score = classes_scores[i][class_id]
                if score > self.score_threshold:
                    boxes.append(np.concatenate([output1[i, :4], np.array([score, class_id])]))
                    scores.append(score)
                    class_ids.append(class_id) 
                    preds.append(output1[i])   
                      
        if len(boxes):        
            boxes = np.array(boxes)
            boxes = xywh2xyxy(boxes)
            scores = np.array(scores)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
            boxes = boxes[indices]
     
            masks_in = np.array(preds)[indices][..., -32:]
            proto= np.squeeze(self.output0).astype(dtype=np.float32)
            c, mh, mw = proto.shape 
            masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)
            
            downsampled_bboxes = boxes.copy()
            downsampled_bboxes[:, 0] *= mw / self.input_shape[0]
            downsampled_bboxes[:, 2] *= mw / self.input_shape[0]
            downsampled_bboxes[:, 3] *= mh / self.input_shape[1]
            downsampled_bboxes[:, 1] *= mh / self.input_shape[1]
        
            masks = crop_mask(masks, downsampled_bboxes)
            if self.draw_result:
                self.result = draw_result(self.image, boxes, masks, self.input_shape)