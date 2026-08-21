'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2026-08-04 00:12:23
Description: __init__.py
'''


from backends.TensorRT.yolo_tensorrt_classify import YOLO_TensorRT_Classify
from backends.TensorRT.yolo_tensorrt_detect import YOLO_TensorRT_Detect
from backends.TensorRT.yolo_tensorrt_segment import YOLO_TensorRT_Segment
from backends.TensorRT.yolo_tensorrt_pose import YOLO_TensorRT_Pose
from backends.TensorRT.yolo_tensorrt_obb import YOLO_TensorRT_OBB
from backends.TensorRT.yolo_tensorrt_depth import YOLO_TensorRT_Depth


__all__ = 'YOLO_TensorRT_Classify', 'YOLO_TensorRT_Detect', 'YOLO_TensorRT_Segment', 'YOLO_TensorRT_Pose', 'YOLO_TensorRT_OBB', 'YOLO_TensorRT_Depth'