'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-08-20 23:31:39
FilePath: \yolo-inference\python\backends\ONNXRuntime\__init__.py
'''


from backends.ONNXRuntime.yolo_onnxruntime import YOLO_ONNXRuntime_Classify, YOLO_ONNXRuntime_Detect, YOLO_ONNXRuntime_Segment


__all__ = "YOLO_ONNXRuntime_Classify", "YOLO_ONNXRuntime_Detect", "YOLO_ONNXRuntime_Segment"