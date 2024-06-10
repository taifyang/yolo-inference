import argparse
import importlib
from yolo import *


def parse_args():
    parser = argparse.ArgumentParser('yolo_inference')
    parser.add_argument('--algo_type', default='YOLOv8', type=str, help='YOLOv5, YOLOv8')
    parser.add_argument('--backend_type', default='TensorRT', type=str, help='ONNXRuntime, OpenCV, OpenVINO, TensorRT')
    parser.add_argument('--task_type', default='Segmentation', type=str, help='Classification, Detection, Segmentation')
    parser.add_argument('--device_type',  default='GPU', type=str, help='CPU, GPU')
    parser.add_argument('--model_type',  default='FP32', type=str, help='FP32, FP16, INT8')
    parser.add_argument('--model_path', default='yolov8n_seg_fp32.engine', type=str, help='the path of model')
    parser.add_argument('--input_path', default="bus.jpg", type=str, help='save result')
    parser.add_argument('--output_path', default="", type=str, help='save result')
    parser.add_argument('--show_result', default=False, type=bool, help='show result')
    parser.add_argument('--save_result', default=True, type=bool, help='save result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    backend_type = args.backend_type
    backend = importlib.import_module('yolo_' + backend_type.lower()) 
    yolo = getattr(backend, 'YOLO_' + backend_type + '_' + args.task_type)
    
    model_path = args.model_path
    
    if args.algo_type == 'YOLOv5':
        algo_type = Algo_Type.YOLOv5
    if args.algo_type == 'YOLOv8':
        algo_type = Algo_Type.YOLOv8
        
    if args.task_type == 'Classification':
        task_type = Task_Type.Classification
    if args.task_type == 'Detection':
        task_type = Task_Type.Detection
    if args.task_type == 'Segmentation':
        task_type = Task_Type.Segmentation  
        
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
    
    show_result = args.show_result and (task_type == Task_Type.Detection or task_type == Task_Type.Segmentation)
    save_result = args.save_result and (task_type == Task_Type.Detection or task_type == Task_Type.Segmentation)
    
    args.output_path = "./result/"+str(args.algo_type)+"_"+str(args.backend_type)+"_"+str(args.task_type)+"_"+str(args.device_type)+"_"+str(args.model_type)+".jpg"
    
    yolo = yolo(algo_type, device_type, model_type, model_path)
    yolo.infer(args.input_path, args.output_path, show_result, save_result)

