'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-10-13 21:54:16
FilePath: \python\backends\TensorRT\yolo_tensorrt.py
Description: tensorrt inference class for YOLO algorithm
'''

import tensorrt as trt
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
            assert runtime, 'runtime create failed!'
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine, 'engine create failed!'
        self.context = self.engine.create_execution_context()
        assert self.context, 'context create failed!'

        self.input_shapes = [] 
        self.outputs_shape = [] 
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(name)
                self.input_shapes.append(shape)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.engine.get_tensor_shape(name)
                self.outputs_shape.append(shape)


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
        assert self.algo_type in ['YOLOv5', 'YOLOv8', 'YOLOv11', 'YOLOv12'], 'algo type not supported!'
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output_ptr = self.output0_device.data.ptr
    
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
            crop_image = cupy.asarray(self.image)[top:(top+crop_size), left:(left+crop_size), ...]
            zoom_factors = (self.inputs_shape[0]/crop_image.shape[0], self.inputs_shape[1]/crop_image.shape[1], 1) 
            input = ndimage.zoom(crop_image, zoom_factors, order=0)
            input = input.astype(cupy.float32) / 255.0
            input = input - cupy.asarray([0.406, 0.456, 0.485], dtype=cupy.float32).reshape(1, 1, -1)
            input = input / cupy.asarray([0.225, 0.224, 0.229], dtype=cupy.float32).reshape(1, 1, -1)
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                zoom_factors = (self.inputs_shape[0]/self.image.shape[0], self.inputs_shape[0]/self.image.shape[0], 1) 
            else:
                zoom_factors = (self.inputs_shape[1]/self.image.shape[1], self.inputs_shape[1]/self.image.shape[1], 1) 
            input = ndimage.zoom(cupy.asarray(self.image), zoom_factors, order=0)
            crop_size = min(input.shape[0], input.shape[1])
            left = (input.shape[1] - crop_size) // 2
            top = (input.shape[0] - crop_size) // 2
            input = input[top:(top+crop_size), left:(left+crop_size), ...]
            input = input.astype(cupy.float32) / 255.0
      
        input = cupy.transpose(input[:, :, ::-1], (2, 0, 1))
        input = cupy.ascontiguousarray(input) 
        self.input_ptr = input.data.ptr

    '''
    description:    model inference
    param {*} self  instance of class
    return {*}
    '''       
    def process(self) -> None:
        self.context.execute_v2(bindings=[self.input_ptr, self.output_ptr])
        self.output0_host = cupy.asnumpy(self.output0_device) 

    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''          
    def post_process(self) -> None:
        output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
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
        assert self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13'], 'algo type not supported!'
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output0_ptr = self.output0_device.data.ptr

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''       
    def pre_process(self) -> None:
        # input = letterbox(self.image, self.inputs_shape)
        # input = np.transpose(input[:, :, ::-1], (2, 0, 1))
        # input = cupy.asarray(input).astype(cupy.float32) / 255.0
        input = letterbox_cupy(self.image, self.inputs_shape)
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
        class_ids = []
        output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))

        if self.algo_type in ['YOLOv3', 'YOLOv4', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11', 'YOLOv12', 'YOLOv13']: 
            classes_scores = output[..., 4:(4 + self.class_num)]  
            class_ids = np.argmax(classes_scores, axis=-1)  
            scores_all = np.max(classes_scores, axis=-1)        
            mask = scores_all > self.score_threshold  
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()   
        elif self.algo_type in ['YOLOv5', 'YOLOv7']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5 + self.class_num)]
            class_ids = np.argmax(classes_scores, axis=-1)
            class_scores = np.max(classes_scores, axis=-1)
            scores_all = class_scores * output[..., 4]        
            mask = scores_all > self.score_threshold
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze() 
          
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
        input = letterbox_cupy(self.image, self.inputs_shape)
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
        else:
            output = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
        boxes = []
        scores = []
        class_ids = []
        preds = []
        
        if self.algo_type in ['YOLOv5']:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:(5 + self.class_num)]
            class_ids = np.argmax(classes_scores, axis=-1)
            class_scores = np.max(classes_scores, axis=-1)
            scores_all = class_scores * output[..., 4]        
            mask = scores_all > self.score_threshold
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()    
            preds = output[mask]           
        elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']: 
            classes_scores = output[..., 4:(4 + self.class_num)]  
            class_ids = np.argmax(classes_scores, axis=-1)  
            scores_all = np.max(classes_scores, axis=-1)        
            mask = scores_all > self.score_threshold  
            boxes = output[mask, :4] 
            scores = scores_all[mask, None]  
            class_ids = class_ids[mask, None]  
            boxes = np.hstack([boxes, scores, class_ids])
            scores = scores.squeeze()     
            preds = output[mask]                        
                      
        if len(boxes):   
            boxes = np.array(boxes)
            boxes = xywh2xyxy(boxes)
            scores = np.array(scores)
            indices = nms(boxes, scores, self.score_threshold, self.nms_threshold)
            boxes = boxes[indices]          
            masks_in = np.array(preds)[indices][..., -32:]
            if int(trt.__version__.split(".")[0]) < 10:
                proto = np.squeeze(self.output0_host.reshape(self.outputs_shape[0]))
            else:
                proto = np.squeeze(self.output1_host.reshape(self.outputs_shape[1]))
            c, mh, mw = proto.shape 
            if self.algo_type in ['YOLOv5']:
                masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)  
            elif self.algo_type in ['YOLOv8', 'YOLOv9', 'YOLOv11', 'YOLOv12']:
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
                