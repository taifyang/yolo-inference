'''
Author: taifyang 
Date: 2024-07-11 23:48:57
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-08-03 21:00:27
FilePath: \yolo-inference\python\backends\OpenCV\__init__.py
'''


from backends.OpenCV.yolo_opencv import YOLO_OpenCV_Classify, YOLO_OpenCV_Detect, YOLO_OpenCV_Segment


__all__ = 'YOLO_OpenCV_Classify', 'YOLO_OpenCV_Detect', 'YOLO_OpenCV_Segment'