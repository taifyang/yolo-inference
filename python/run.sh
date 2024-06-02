python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type CPU --model_type FP32 --model_path yolov5n_fp32.onnx
python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type GPU --model_type FP32 --model_path yolov5n_fp32.onnx
python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type CPU --model_type FP16 --model_path yolov5n_fp16.onnx 
python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type GPU --model_type FP16 --model_path yolov5n_fp16.onnx
python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type CPU --model_type INT8 --model_path yolov5n_int8.onnx
python main.py --algo_type YOLOv5 --backend_type ONNXRuntime --device_type GPU --model_type INT8 --model_path yolov5n_int8.onnx

python main.py --algo_type YOLOv5 --backend_type OpenCV --device_type CPU --model_type FP32 --model_path yolov5n_fp32.onnx
python main.py --algo_type YOLOv5 --backend_type OpenCV --device_type GPU --model_type FP32 --model_path yolov5n_fp32.onnx
python main.py --algo_type YOLOv5 --backend_type OpenCV --device_type CPU --model_type FP16 --model_path yolov5n_fp16.onnx
python main.py --algo_type YOLOv5 --backend_type OpenCV --device_type GPU --model_type FP16 --model_path yolov5n_fp16.onnx

python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type CPU --model_type FP32 --model_path yolov5n_fp32.xml
# python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type GPU --model_type FP32 --model_path yolov5n_fp32.xml
python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type CPU --model_type FP16 --model_path yolov5n_fp16.xml
# python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type GPU --model_type FP16 --model_path yolov5n_fp16.xml
python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type CPU --model_type INT8 --model_path yolov5n_int8.xml
# python main.py --algo_type YOLOv5 --backend_type OpenVINO --device_type GPU --model_type INT8 --model_path yolov5n_int8.xml

python main.py --algo_type YOLOv5 --backend_type TensorRT --device_type GPU --model_type FP32 --model_path yolov5n_fp32.engine
python main.py --algo_type YOLOv5 --backend_type TensorRT --device_type GPU --model_type FP16 --model_path yolov5n_fp16.engine
python main.py --algo_type YOLOv5 --backend_type TensorRT --device_type GPU --model_type INT8 --model_path yolov5n_int8.engine


python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type CPU --model_type FP32 --model_path yolov8n_fp32.onnx
python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type GPU --model_type FP32 --model_path yolov8n_fp32.onnx
python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type CPU --model_type FP16 --model_path yolov8n_fp16.onnx 
python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type GPU --model_type FP16 --model_path yolov8n_fp16.onnx
python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type CPU --model_type INT8 --model_path yolov8n_int8.onnx
python main.py --algo_type YOLOv8 --backend_type ONNXRuntime --device_type GPU --model_type INT8 --model_path yolov8n_int8.onnx

python main.py --algo_type YOLOv8 --backend_type OpenCV --device_type CPU --model_type FP32 --model_path yolov8n_fp32.onnx
# python main.py --algo_type YOLOv8 --backend_type OpenCV --device_type GPU --model_type FP32 --model_path yolov8n_fp32.onnx
python main.py --algo_type YOLOv8 --backend_type OpenCV --device_type CPU --model_type FP16 --model_path yolov8n_fp16.onnx
# python main.py --algo_type YOLOv8 --backend_type OpenCV --device_type GPU --model_type FP16 --model_path yolov8n_fp16.onnx

python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type CPU --model_type FP32 --model_path yolov8n_fp32.xml
# python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type GPU --model_type FP32 --model_path yolov8n_fp32.xml
python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type CPU --model_type FP16 --model_path yolov8n_fp16.xml
# python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type GPU --model_type FP16 --model_path yolov8n_fp16.xml
python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type CPU --model_type INT8 --model_path yolov8n_int8.xml
# python main.py --algo_type YOLOv8 --backend_type OpenVINO --device_type GPU --model_type INT8 --model_path yolov8n_int8.xml

python main.py --algo_type YOLOv8 --backend_type TensorRT --device_type GPU --model_type FP32 --model_path yolov8n_fp32.engine
python main.py --algo_type YOLOv8 --backend_type TensorRT --device_type GPU --model_type FP16 --model_path yolov8n_fp16.engine
python main.py --algo_type YOLOv8 --backend_type TensorRT --device_type GPU --model_type INT8 --model_path yolov8n_int8.engine