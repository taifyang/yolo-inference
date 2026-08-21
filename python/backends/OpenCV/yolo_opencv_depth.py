'''
Author: taifyang  
Date: 2024-06-12 22:23:07
LastEditTime: 2026-08-03 23:54:52
Description: opencv inference class for YOLO depth estimation algorithm
'''


from backends.utils import *
from backends.OpenCV.yolo_opencv import *


'''
description: opencv inference class for the YOLO depth estimation algorithm
'''    
class YOLO_OpenCV_Depth(YOLO_OpenCV):
    '''
    description:    model pre-process
    param {*} self  instance of class
    return {*}
    ''' 
    def pre_process(self) -> None:
        assert self.algo_type in ['YOLO26'], 'algo type not supported!'
        self.inputs_shape = (768, 768)
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
        depth = scale_mask(output, self.inputs_shape, self.image.shape)
        if self.draw_result:
            self.result = draw_result(depth)