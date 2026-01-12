'''
Author: taifyang
Date: 2024-06-12 22:23:07
LastEditTime: 2025-12-26 22:06:14
Description: openvino inference class for the YOLO classifaction algorithm
'''


from backends.utils import *
from backends.OpenVINO.yolo_openvino import YOLO_OpenVINO


'''
description: openvino inference class for the YOLO classifaction algorithm
'''
class YOLO_OpenVINO_Classify(YOLO_OpenVINO):
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
        self.inputs = np.expand_dims(input, axis=0)
    
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output = self.outputs[self.compiled_model.output(0)]
        output = np.squeeze(output).astype(dtype=np.float32)
        if self.algo_type in ['YOLOv5'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        elif self.algo_type in ['YOLOv8', 'YOLOv11', 'YOLOv12'] and self.draw_result:
            print('class:', np.argmax(output), ' scores:', np.max(output))
       
