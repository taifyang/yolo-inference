'''
Author: taifyang 58515915+taifyang@users.noreply.github.com
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-06-29 20:26:01
FilePath: \python\backends\yolo.py
Description: YOLO接口类
'''

import os
import cv2
import time
import backends
    

'''
description: YOLO接口类
'''
class YOLO:  
    '''
    description:    构造方法
    param {*} self  类的实例
    return {*}
    '''    
    def __init__(self) -> None:
        super().__init__()
        self.score_threshold = 0.2      #得分阈值
        self.nms_threshold = 0.5        #NMS阈值
        self.confidence_threshold = 0.2 #置信度阈值    
        self.input_shape = (640, 640)   #输入图像尺寸
    
    '''
    description:    任务映射表
    param {*} self
    return {*}      算法类实例
    '''    
    def task_map(self):
        return {
            'ONNXRuntime':{
                'Classify':backends.ONNXRuntime.YOLO_ONNXRuntime_Classify,
                'Detect':backends.ONNXRuntime.YOLO_ONNXRuntime_Detect,
                'Segment':backends.ONNXRuntime.YOLO_ONNXRuntime_Segment,
            },
            'OpenCV':{
                'Classify':backends.OpenCV.YOLO_OpenCV_Classify,
                'Detect':backends.OpenCV.YOLO_OpenCV_Detect,
                #'Segment':tasks.OpenCV.YOLO_OpenCV_Segment,
            },
            'OpenVINO':{
                'Classify':backends.OpenVINO.YOLO_OpenVINO_Classify,
                'Detect':backends.OpenVINO.YOLO_OpenVINO_Detect,
                'Segment':backends.OpenVINO.YOLO_OpenVINO_Segment,
            },
            'TensorRT':{
                'Classify':backends.TensorRT.YOLO_TensorRT_Classify,
                'Detect':backends.TensorRT.YOLO_TensorRT_Detect,
                'Segment':backends.TensorRT.YOLO_TensorRT_Segment,
            },
        }
    
    '''
    description:                推理接口
    param {*} self              类的实例
    param {str} input_path      输入路径
    param {str} output_path     输出路径
    param {bool} save_result    保存结果
    param {bool} show_result    显示结果
    return {*}
    '''    
    def infer(self, input_path:str, output_path:str, save_result:bool, show_result:bool) -> None:
        assert os.path.exists(input_path), 'input not exists!'
        self.draw_result = save_result or show_result
        if input_path.endswith('.bmp') or input_path.endswith('.jpg') or input_path.endswith('.png'):
            self.image = cv2.imread(input_path)
            self.result = self.image.copy()
            
            for i in range(10):
                start = time.perf_counter()
                self.pre_process()
                self.process()
                self.post_process()
                end = time.perf_counter()
                print((end-start)*1000, 'ms')  
            
            if save_result and output_path!='':
                cv2.imwrite(output_path, self.result)
            if show_result:
                cv2.imshow('result', self.result)
                cv2.waitKey(0)
        elif input_path.endswith('.mp4'):
            cap = cv2.VideoCapture(input_path)
            start = time.perf_counter()
            if save_result and output_path!='':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                wri = cv2.VideoWriter(output_path, fourcc, 30.0, (1280,720))
            while True:
                ret, self.image  = cap.read()
                if not ret:
                    break
                self.result = self.image.copy()
                self.pre_process()
                self.process()
                self.post_process()
                if show_result:
                    cv2.imshow('result', self.result)
                    cv2.waitKey(1)
                if save_result and output_path!='':
                    wri.write(self.result)
            end = time.perf_counter()
            print((end-start)*1000, 'ms')                         
            
    