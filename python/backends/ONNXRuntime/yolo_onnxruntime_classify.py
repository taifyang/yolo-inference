'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-23 08:38:15
Description: onnxruntime inference class for YOLO classifaction algorithm
'''


from backends.utils import *
from backends.ONNXRuntime.yolo_onnxruntime import *


'''
description: onnxruntime inference class for the YOLO classifaction algorithm
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
            input = centercrop(self.image, self.inputs_shape)
            input = normalize(input, self.algo_type)
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                self.image = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                self.image = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            input = centercrop(self.image, self.inputs_shape)
            input = normalize(input, self.algo_type)
            
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

