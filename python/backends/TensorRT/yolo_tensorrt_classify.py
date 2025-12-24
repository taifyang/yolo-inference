'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:37:14
Description: tensorrt inference class for YOLO classifaction algorithm
'''


from backends.utils import *
from backends.TensorRT.yolo_tensorrt import *


'''
description: tensorrt inference class for the YOLO classifaction algorithm
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
    