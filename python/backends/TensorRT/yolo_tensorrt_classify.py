'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 00:16:24
Description: tensorrt inference class for YOLO classifaction algorithm
'''


from backends.utils import *
from backends.yolo_classify import *
from backends.TensorRT.yolo_tensorrt import *


'''
description: tensorrt inference class for the YOLO classifaction algorithm
'''        
class YOLO_TensorRT_Classify(YOLO_TensorRT, YOLO_Classify):
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
        YOLO_TensorRT.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Classify.__init__(self, algo_type, device_type, model_type, model_path)
        self.output0_device = cupy.empty(self.outputs_shape[0], dtype=np.float32)
        self.output_ptr = self.output0_device.data.ptr
    
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''       
    def pre_process(self) -> None:
        if self.algo_type in ['YOLOv5']:
            input = centercrop(self.image, self.inputs_shape, use_cupy=True)
            input = normalize(input, self.algo_type, use_cupy=True)
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                input = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                input = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            input = centercrop(input, self.inputs_shape, use_cupy=True)
            input = normalize(input, self.algo_type, use_cupy=True)
      
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
        if self.algo_type in ['YOLOv5']:
            class_id = np.argmax(output)
            scores =np.exp(np.max(output))/np.sum(np.exp(output))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            class_id = np.argmax(output)
            scores = np.max(output)

        if self.render_result:
            self.result = self.draw_result(class_id, scores)