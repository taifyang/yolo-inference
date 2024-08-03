'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-08-03 21:02:51
FilePath: \yolo-inference\python\backends\__init__.py
'''
from backends import ONNXRuntime, OpenCV, OpenVINO, TensorRT


__all__ = 'ONNXRuntime', 'OpenCV', 'OpenVINO', 'TensorRT'