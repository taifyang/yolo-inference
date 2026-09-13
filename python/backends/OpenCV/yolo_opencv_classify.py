'''
Author: taifyang
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 00:14:57
Description: opencv inference class for the YOLO classifaction algorithm
'''


from backends.utils import *
from backends.yolo_classify import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO classifaction algorithm
'''     
class YOLO_OpenCV_Classify(YOLO_OpenCV, YOLO_Classify):
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
        YOLO_OpenCV.__init__(self, algo_type, device_type, model_type, model_path)
        YOLO_Classify.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    '''    
    def pre_process(self) -> None:
        if self.algo_type in ['YOLOv5']:
            input = centercrop(self.image, self.inputs_shape)
            input = normalize(input, self.algo_type)
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            self.inputs_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                input = cv2.resize(self.image, (self.inputs_shape[0]*self.image.shape[1]//self.image.shape[0], self.inputs_shape[0]))
            else:
                input = cv2.resize(self.image, (self.inputs_shape[1], self.inputs_shape[1]*self.image.shape[0]//self.image.shape[1]))
            input = centercrop(input, self.inputs_shape)
            input = normalize(input, self.algo_type)
            
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB and HWC2CHW
        self.inputs = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        self.net.setInput(self.inputs)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''    
    def post_process(self) -> None:
        output = np.squeeze(self.outputs).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5']:
            class_id = np.argmax(output)
            scores =np.exp(np.max(output))/np.sum(np.exp(output))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12', 'YOLO26']:
            class_id = np.argmax(output)
            scores = np.max(output)

        if self.render_result:
            self.result = self.draw_result(class_id, scores)
