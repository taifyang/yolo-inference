'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-08-23 14:34:47
Description: opencv inference class for YOLO semantic segmentation algorithm
'''


from backends.utils import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO semantic segmentation algorithm
'''    
class YOLO_OpenCV_Semantic(YOLO_OpenCV):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (1024, 1024)
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
        if self.draw_result:
            masks = []
            for cls_id in np.unique(mask):
                binary_mask = (mask == cls_id).astype(np.bool_)
                masks.append(binary_mask)
            self.result = draw_result(task_type='Semantic', image=self.image, masks=masks)