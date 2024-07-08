'''
Author: taifyang 58515915+taifyang@users.noreply.github.com
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-07-07 13:38:50
Description: 调用demo
'''
import argparse
from backends.yolo import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_type', default='YOLOv6', type=str, help='YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10')
    parser.add_argument('--backend_type', default='TensorRT', type=str, help='ONNXRuntime, OpenCV, OpenVINO, TensorRT')
    parser.add_argument('--task_type', default='Detect', type=str, help='Classify, Detect, Segment')
    parser.add_argument('--device_type',  default='GPU', type=str, help='CPU, GPU')
    parser.add_argument('--model_type',  default='INT8', type=str, help='FP32, FP16, INT8')
    parser.add_argument('--model_path', default='../weights/yolov6n_int8.engine', type=str, help='the path of model')
    parser.add_argument('--input_path', default='bus.jpg', type=str, help='save result')
    parser.add_argument('--output_path', default='', type=str, help='save result')
    parser.add_argument('--save_result', default=True, type=bool, help='save result')
    parser.add_argument('--show_result', default=True, type=bool, help='show result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.show_result = args.show_result and (args.task_type == 'Detect' or args.task_type == 'Segment')
    args.save_result = args.save_result and (args.task_type == 'Detect' or args.task_type == 'Segment')
    args.output_path = './result/'+str(args.algo_type)+'_'+str(args.backend_type)+'_'+str(args.task_type)+'_'+str(args.device_type)+'_'+str(args.model_type)+'.jpg'
    
    yolo = YOLO().task_map()[args.backend_type][args.task_type] 
    yolo = yolo(args.algo_type, args.device_type, args.model_type, args.model_path)
    yolo.infer(args.input_path, args.output_path, args.save_result, args.show_result)