'''
Author: taifyang
Date: 2025-11-28 20:35:27
LastEditTime: 2026-08-23 12:50:57
Description: __init__.py
'''


from backends.OpenCV.yolo_opencv_classify import YOLO_OpenCV_Classify
from backends.OpenCV.yolo_opencv_detect import YOLO_OpenCV_Detect
from backends.OpenCV.yolo_opencv_segment import YOLO_OpenCV_Segment
from backends.OpenCV.yolo_opencv_pose import YOLO_OpenCV_Pose
from backends.OpenCV.yolo_opencv_obb import YOLO_OpenCV_OBB
from backends.OpenCV.yolo_opencv_depth import YOLO_OpenCV_Depth
from backends.OpenCV.yolo_opencv_semantic import YOLO_OpenCV_Semantic


__all__ = tuple(k for k in dir() if k.startswith("YOLO_OpenCV_"))