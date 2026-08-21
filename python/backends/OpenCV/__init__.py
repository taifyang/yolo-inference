'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2026-08-03 23:54:38
Description: __init__.py
'''


from backends.OpenCV.yolo_opencv_classify import YOLO_OpenCV_Classify
from backends.OpenCV.yolo_opencv_detect import YOLO_OpenCV_Detect
from backends.OpenCV.yolo_opencv_segment import YOLO_OpenCV_Segment
from backends.OpenCV.yolo_opencv_pose import YOLO_OpenCV_Pose
from backends.OpenCV.yolo_opencv_obb import YOLO_OpenCV_OBB
from backends.OpenCV.yolo_opencv_depth import YOLO_OpenCV_Depth


__all__ = 'YOLO_OpenCV_Classify', 'YOLO_OpenCV_Detect', 'YOLO_OpenCV_Segment', 'YOLO_OpenCV_Pose', 'YOLO_OpenCV_OBB', 'YOLO_OpenCV_Depth'