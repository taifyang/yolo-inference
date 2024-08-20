'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 
LastEditTime: 2024-08-03 21:02:23
FilePath: \yolo-inference\python\backends\OpenVINO\__init__.py
'''


from backends.OpenVINO.yolo_openvino import YOLO_OpenVINO_Classify, YOLO_OpenVINO_Detect, YOLO_OpenVINO_Segment


__all__ = 'YOLO_OpenVINO_Classify', 'YOLO_OpenVINO_Detect', 'YOLO_OpenVINO_Segment'