'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2026-08-04 00:05:56
Description: __init__.py
'''


from backends.OpenVINO.yolo_openvino_classify import YOLO_OpenVINO_Classify
from backends.OpenVINO.yolo_openvino_detect import YOLO_OpenVINO_Detect
from backends.OpenVINO.yolo_openvino_segment import YOLO_OpenVINO_Segment
from backends.OpenVINO.yolo_openvino_pose import YOLO_OpenVINO_Pose
from backends.OpenVINO.yolo_openvino_obb import YOLO_OpenVINO_OBB
from backends.OpenVINO.yolo_openvino_depth import YOLO_OpenVINO_Depth


__all__ = 'YOLO_OpenVINO_Classify', 'YOLO_OpenVINO_Detect', 'YOLO_OpenVINO_Segment', 'YOLO_OpenVINO_Pose', 'YOLO_OpenVINO_OBB', 'YOLO_OpenVINO_Depth'