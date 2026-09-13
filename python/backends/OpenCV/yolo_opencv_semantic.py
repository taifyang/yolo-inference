'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-09-11 21:49:02
Description: opencv inference class for YOLO semantic segmentation algorithm
'''


from backends.utils import *
from backends.yolo_semantic import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO semantic segmentation algorithm
'''    
class YOLO_OpenCV_Semantic(YOLO_OpenCV, YOLO_Semantic):
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
        YOLO_Semantic.__init__(self, algo_type, device_type, model_type, model_path)

    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        input = letterbox(self.image, self.inputs_shape)
        self.inputs = cv2.dnn.blobFromImage(input, 1/255., size=self.inputs_shape, swapRB=True, crop=False)
        self.net.setInput(self.inputs)
        
    '''
    description:    model post-process
    param {*} self  instance of class
    return {*}
    '''           
    def post_process(self) -> None:
        output = np.squeeze(self.outputs[0]).astype(dtype=np.float32)
        class_map = scale_masks(output, self.inputs_shape, self.image.shape)
        mask = class_map.astype(np.uint8)
        if self.render_result:
            masks = []
            for cls_id in np.unique(mask):
                binary_mask = (mask == cls_id).astype(np.bool_)
                masks.append(binary_mask)
            self.result = self.draw_result(masks)