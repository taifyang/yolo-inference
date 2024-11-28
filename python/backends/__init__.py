'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-11-07 21:14:54
FilePath: \yolo-inference\python\backends\__init__.py
'''


try:
    from backends import ONNXRuntime
except:
    print('ONNXRuntime module import fail!')
    
try:
    from backends import OpenCV
except:
    print('OpenCV module import fail!')
   
try:
    from backends import OpenVINO
except:
    print('OpenVINO module import fail!')

try:
    from backends import PyTorch
except:
    print('PyTorch module import fail!')
            
try:
    from backends import TensorRT
except:
    print('TensorRT module import fail!')


__all__ = 'ONNXRuntime', 'OpenCV', 'OpenVINO', 'PyTorch', 'TensorRT'
