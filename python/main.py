import argparse
import importlib
from yolov5 import *


def parse_args():
    parser = argparse.ArgumentParser('yolov5')
    parser.add_argument('--algo_type', default='ONNXRuntime', type=str, help='ONNXRuntime, OpenCV, OpenVINO, TensorRT')
    parser.add_argument('--model_path', default='yolov5n_fp32.onnx', type=str, help='the path of model')
    parser.add_argument('--device_type',  default='cpu', type=str, help='cpu, gpu')
    parser.add_argument('--model_type',  default='fp32', type=str, help='fp32, fp16, int8')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    algo_type = args.algo_type
    algo = importlib.import_module('yolov5_' + algo_type.lower()) 
    YOLOv5 = getattr(algo, 'YOLOv5_' + algo_type)
    
    model_path = args.model_path
    
    if args.device_type == 'cpu':
        device_type = Device_Type.CPU
    elif args.device_type == 'gpu':
        device_type = Device_Type.GPU
        
    if args.model_type == 'fp32':
        model_type = Model_Type.FP32
    elif args.model_type == 'fp16':
        model_type = Model_Type.FP16
    elif args.model_type == 'int8':
        model_type = Model_Type.INT8
        
    yolov5 = YOLOv5(model_path, device_type, model_type)
    yolov5.infer("test.mp4")
