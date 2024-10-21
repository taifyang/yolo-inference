'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang
LastEditTime: 2024-10-21 23:01:19
FilePath: \yolo-inference\python\backends\PyTorch\__init__.py
'''


from backends.PyTorch.yolo_pytorch import YOLO_PyTorch_Classify, YOLO_PyTorch_Detect, YOLO_PyTorch_Segment


__all__ = 'YOLO_PyTorch_Classify', 'YOLO_PyTorch_Detect', 'YOLO_PyTorch_Segment'