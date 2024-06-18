'''
Author: taifyang 58515915+taifyang@users.noreply.github.com
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-06-18 21:54:07
Description:  yolo算法opencv推理框架实现
'''
import cv2
from backends.yolo import *
from backends.utils import *


'''
description: yolo算法opencv推理框架实现类
'''
class YOLO_OpenCV(YOLO):
    '''
    description:            构造方法
    param {*} self
    param {str} algo_type   算法类型
    param {str} device_type 设备类型
    param {str} model_type  模型精度
    param {str} model_path  模型路径
    return {*}
    '''    
    def __init__(self, algo_type:str, device_type:str, model_type:str, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), 'model not exists!'
        self.net = cv2.dnn.readNet(model_path)
        self.algo_type = algo_type
        if device_type == 'GPU':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    '''
    description:    模型推理
    param {*} self
    return {*}
    '''       
    def process(self) -> None:
        self.output = self.net.forward()

'''
description: yolo分类算法opencv推理框架实现类
'''     
class YOLO_OpenCV_Classify(YOLO_OpenCV):
    '''
    description:    模型前处理
    param {*} self
    return {*}
    '''    
    def pre_process(self) -> None:
        if self.algo_type == 'YOLOv5':
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
            input = input - np.array([0.406, 0.456, 0.485])
            input = input / np.array([0.225, 0.224, 0.229])
        if self.algo_type == 'YOLOv8':
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
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        self.input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        self.net.setInput(self.input)
    
    '''
    description:    模型后处理
    param {*} self
    return {*}
    '''    
    def post_process(self) -> None:
        output = np.squeeze(self.output).astype(dtype=np.float32)
        if self.algo_type == 'YOLOv5':
            print('class:', np.argmax(output), ' scores:', np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type == 'YOLOv8':
            print('class:', np.argmax(output), ' scores:', np.max(output))
    

'''
description: yolo检测算法opencv推理框架实现类
'''      
class YOLO_OpenCV_Detect(YOLO_OpenCV):
    '''
    description:    模型前处理
    param {*} self
    return {*}
    '''    
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        self.input = cv2.dnn.blobFromImage(input, 1/255., size=self.input_shape, swapRB=True, crop=False)
        self.net.setInput(self.input)
    
    '''
    description:    模型后处理
    param {*} self
    return {*}
    '''     
    def post_process(self) -> None:
        boxes = []
        scores = []
        class_ids = []
        for i in range(self.output.shape[1]):
            if self.algo_type == 'YOLOv5':
                data = self.output[0][i]
                objness = data[4]
                if objness < self.confidence_threshold:
                    continue
                score = data[5:] * objness
            if self.algo_type == 'YOLOv8':
                data = self.output[0][i, ...]
                score = data[4:]
                objness = 1
            _, _, _, max_score_index = cv2.minMaxLoc(score)
            max_id = max_score_index[1]
            if score[max_id] > self.score_threshold:
                x, y, w, h = data[0].item(), data[1].item(), data[2].item(), data[3].item()
                boxes.append(np.array([x-w/2, y-h/2, x+w/2, y+h/2]))
                scores.append(score[max_id]*objness)
                class_ids.append(max_id)
        if len(boxes):   
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_threshold)
            output = []
            for i in indices:
                output.append(np.array([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], class_ids[i]]))
            boxes = np.array(output)
            self.result = draw(self.image, boxes)

