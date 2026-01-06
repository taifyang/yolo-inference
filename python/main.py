'''
Author: taifyang 
Date: 2024-06-12 22:23:07
LastEditTime: 2026-01-05 11:25:39
Description: demo
'''

import os
import argparse
from backends.yolo import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_type', default='YOLOv8', type=str, help='YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv13')
    parser.add_argument('--backend_type', default='PyTorch', type=str, help='PyTorch, ONNXRuntime, OpenCV, OpenVINO, TensorRT')
    parser.add_argument('--task_type', default='Pose', type=str, help='Classify, Detect, Segment, Pose')
    parser.add_argument('--device_type',  default='GPU', type=str, help='CPU, GPU')
    parser.add_argument('--model_type',  default='FP16', type=str, help='FP32, FP16, INT8')
    parser.add_argument('--model_path', default='./weights/yolov8n_seg_fp16_gpu.torchscript', type=str, help='the path of model')
    parser.add_argument('--input_path', default='./bus.jpg', type=str, help='the input path of input image or video')
    parser.add_argument('--output_path', default='', type=str, help='the output path of input image or video')
    parser.add_argument('--save_result', default=True, type=bool, help='save result')
    parser.add_argument('--show_result', default=False, type=bool, help='show result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not args.output_path:
        args.output_path = 'result/'+str(args.algo_type)+'_'+str(args.backend_type)+'_'+str(args.task_type)+'_'+str(args.device_type)+'_'+str(args.model_type)+ os.path.splitext(args.input_path)[-1] 
    yolo = YOLO().task_map()[args.backend_type][args.task_type] 
    yolo = yolo(args.algo_type, args.device_type, args.model_type, args.model_path)
    yolo.infer(args.input_path, args.output_path, args.save_result, args.show_result)