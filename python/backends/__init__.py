'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 
LastEditTime: 2024-08-20 23:15:30
FilePath: \yolo-inference\python\backends\__init__.py
'''


try:
    import onnxruntime
    from backends import ONNXRuntime
except:
    print('onnxruntime is not installed')
    
try:
    import cv2
    from backends import OpenCV
except:
    print('cv2 is not installed')
   
try:
    import openvino
    from backends import OpenVINO
except:
    print('openvino is not installed')

try:
    import torch
    from backends import PyTorch
except:
    print('pytorch is not installed')
            
try:
    import tensorrt
    from backends import TensorRT
except:
    print('tensorrt is not installed')


__all__ = 'ONNXRuntime', 'OpenCV', 'OpenVINO', 'PyTorch', 'TensorRT'
