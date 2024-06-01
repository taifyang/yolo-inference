import argparse
import importlib
from yolo import *


def parse_args():
    parser = argparse.ArgumentParser('yolo_inference')
    parser.add_argument('--algo_type', default='YOLOv5', type=str, help='YOLOv5, YOLOv8')
    parser.add_argument('--backend_type', default='ONNXRuntime', type=str, help='ONNXRuntime, OpenCV, OpenVINO, TensorRT')
    parser.add_argument('--device_type',  default='CPU', type=str, help='CPU, GPU')
    parser.add_argument('--model_type',  default='FP32', type=str, help='FP32, FP16, INT8')
    parser.add_argument('--model_path', default='yolov5n_fp32.onnx', type=str, help='the path of model')
    parser.add_argument('--input_path', default="test.mp4", type=str, help='save result')
    parser.add_argument('--output_path', default="result.avi", type=str, help='save result')
    parser.add_argument('--show_result', default=False, type=bool, help='show result')
    parser.add_argument('--save_result', default=False, type=bool, help='save result')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    backend_type = args.backend_type
    backend = importlib.import_module('yolo_' + backend_type.lower()) 
    yolo = getattr(backend, 'YOLO_' + backend_type)
    
    model_path = args.model_path
    
    if args.algo_type == 'YOLOv5':
        algo_type = Algo_Type.YOLOv5
    if args.algo_type == 'YOLOv8':
        algo_type = Algo_Type.YOLOv8
    
    if args.device_type == 'CPU':
        device_type = Device_Type.CPU
    if args.device_type == 'GPU':
        device_type = Device_Type.GPU
        
    if args.model_type == 'FP32':
        model_type = Model_Type.FP32
    if args.model_type == 'FP16':
        model_type = Model_Type.FP16
    if args.model_type == 'INT8':
        model_type = Model_Type.INT8
    
    yolo = yolo(algo_type, device_type, model_type, model_path)
    yolo.infer(args.input_path, args.output_path, args.show_result, args.save_result)

