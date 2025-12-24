'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2024-08-20 23:29:53
FilePath: \yolo-inference\python\backends\TensorRT\__init__.py
'''


from backends.TensorRT.yolo_tensorrt import YOLO_TensorRT_Classify, YOLO_TensorRT_Detect, YOLO_TensorRT_Segment


__all__ = 'YOLO_TensorRT_Classify', 'YOLO_TensorRT_Detect', 'YOLO_TensorRT_Segment'