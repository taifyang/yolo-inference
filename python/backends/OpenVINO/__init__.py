'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2024-08-03 21:02:23
Description: __init__.py
'''


from backends.OpenVINO.yolo_openvino_classify import YOLO_OpenVINO_Classify
from backends.OpenVINO.yolo_openvino_detect import YOLO_OpenVINO_Detect
from backends.OpenVINO.yolo_openvino_segment import YOLO_OpenVINO_Segment


__all__ = 'YOLO_OpenVINO_Classify', 'YOLO_OpenVINO_Detect', 'YOLO_OpenVINO_Segment'