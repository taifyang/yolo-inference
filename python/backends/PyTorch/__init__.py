'''
Author: taifyang
Date: 2025-11-28 20:35:27
LastEditTime: 2026-08-23 12:50:22
Description: __init__.py
'''


from backends.PyTorch.yolo_pytorch_classify import YOLO_PyTorch_Classify
from backends.PyTorch.yolo_pytorch_detect import YOLO_PyTorch_Detect
from backends.PyTorch.yolo_pytorch_segment import YOLO_PyTorch_Segment
from backends.PyTorch.yolo_pytorch_pose import YOLO_PyTorch_Pose
from backends.PyTorch.yolo_pytorch_obb import YOLO_PyTorch_OBB
from backends.PyTorch.yolo_pytorch_depth import YOLO_PyTorch_Depth
from backends.PyTorch.yolo_pytorch_semantic import YOLO_PyTorch_Semantic


__all__ = tuple(k for k in dir() if k.startswith("YOLO_PyTorch_"))