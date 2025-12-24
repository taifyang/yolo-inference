'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2025-12-22 23:32:15
Description: __init__.py
'''


from backends.OpenCV.yolo_opencv_classify import YOLO_OpenCV_Classify
from backends.OpenCV.yolo_opencv_detect import YOLO_OpenCV_Detect
from backends.OpenCV.yolo_opencv_segment import YOLO_OpenCV_Segment


__all__ = 'YOLO_OpenCV_Classify', 'YOLO_OpenCV_Detect', 'YOLO_OpenCV_Segment'