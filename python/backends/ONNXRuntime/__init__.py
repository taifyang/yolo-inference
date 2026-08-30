'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditTime: 2026-08-23 10:00:22
Description: __init__.py
'''


from backends.ONNXRuntime.yolo_onnxruntime_classify import YOLO_ONNXRuntime_Classify
from backends.ONNXRuntime.yolo_onnxruntime_detect import YOLO_ONNXRuntime_Detect
from backends.ONNXRuntime.yolo_onnxruntime_segment import YOLO_ONNXRuntime_Segment
from backends.ONNXRuntime.yolo_onnxruntime_pose import YOLO_ONNXRuntime_Pose
from backends.ONNXRuntime.yolo_onnxruntime_obb import YOLO_ONNXRuntime_OBB
from backends.ONNXRuntime.yolo_onnxruntime_depth import YOLO_ONNXRuntime_Depth
from backends.ONNXRuntime.yolo_onnxruntime_semantic import YOLO_ONNXRuntime_Semantic


__all__ = tuple(k for k in dir() if k.startswith("YOLO_ONNXRuntime_"))