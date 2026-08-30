'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2026-08-23 14:38:31
Description: __init__.py
'''


from backends.OpenVINO.yolo_openvino_classify import YOLO_OpenVINO_Classify
from backends.OpenVINO.yolo_openvino_detect import YOLO_OpenVINO_Detect
from backends.OpenVINO.yolo_openvino_segment import YOLO_OpenVINO_Segment
from backends.OpenVINO.yolo_openvino_pose import YOLO_OpenVINO_Pose
from backends.OpenVINO.yolo_openvino_obb import YOLO_OpenVINO_OBB
from backends.OpenVINO.yolo_openvino_depth import YOLO_OpenVINO_Depth
from backends.OpenVINO.yolo_openvino_semantic import YOLO_OpenVINO_Semantic


__all__ = tuple(k for k in dir() if k.startswith("YOLO_OpenVINO_"))