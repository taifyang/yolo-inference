# yolo-inference
C++ and Python implementations of YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11 inference.

Supported inference backends include Libtorch/PyTorch, ONNXRuntime, OpenCV, OpenVINO and TensorRT. 

Supported task types include Classify, Detect and Segment.

Supported model types include FP32, FP16 and INT8.

You can test C++ code with:
```powershell
# Windows
mkdir build ; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./run.bat
```
or
```bash
# Linux
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./run.sh
```

C++ test on Ubuntu22.04 in Docker(CPU i7-12700, GPU RTX3070): 
|       Model       |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :------: | :---------: | :----: | :------: | :------: |
| YOLOv5n | Classify | CPU | FP32 | 15.3ms | 15.5ms | 20.6ms | 14.1ms | ×
| YOLOv5n | Classify | GPU | FP32 | 4.9ms | 5.1ms | 5.1ms | ? | 4.1ms
| YOLOv5n | Classify | CPU | FP16 | × | 22.7ms | 20.1ms | 14.0ms | ×
| YOLOv5n | Classify | GPU | FP16 | 4.6ms | 7.7ms | 4.9ms | ? | 3.2ms
| YOLOv5n | Classify | CPU | INT8 | × | 18.9ms | × | ? | ×
| YOLOv5n | Classify | GPU | INT8 | × | 38.9ms | × | ? | 3.0ms
| YOLOv5n | Detect | CPU | FP32 | 23.3ms | 22.0ms | 57.3ms | 20.0ms | ×
| YOLOv5n | Detect | GPU | FP32 | 7.2ms | 6.6ms | 8.2ms | ? | 4.4ms
| YOLOv5n | Detect | CPU | FP16 | × | 42.6ms | 57.3ms | 19.8ms | ×
| YOLOv5n | Detect | GPU | FP16 | 6.8ms | 18.4ms | 7.9ms | ? | 3.9ms
| YOLOv5n | Detect | CPU | INT8 | × | 27.0ms | × | 18.1ms | ×
| YOLOv5n | Detect | GPU | INT8 | × | 51.2ms | × | ? | 3.5ms
| YOLOv5n | Segment | CPU | FP32 | × | 30.7ms | 75.8ms | 27.2ms | ×
| YOLOv5n | Segment | GPU | FP32 | 10.1ms | 10.6ms | 10.8ms | ? | 6.3ms
| YOLOv5n | Segment | CPU | FP16 | × | 55.0ms | 75.9ms | 27.2ms | ×
| YOLOv5n | Segment | GPU | FP16 | 9.8ms | 28.4ms | 10.0ms | ? | 5.0ms
| YOLOv5n | Segment | CPU | INT8 | × | 35.6ms | × | ? | ×
| YOLOv5n | Segment | GPU | INT8 | × | 63.8ms | × | ? | 4.2ms
| YOLOv8n | Classify | CPU | FP32 | 3.5ms | 2.4ms | 4.0ms | 2.4ms | ×
| YOLOv8n | Classify | GPU | FP32 | 2.3ms | 1.5ms | 1.9ms | ? | 1.2ms
| YOLOv8n | Classify | CPU | FP16 | × | 6.4ms | 4.0ms | 2.4ms | ×
| YOLOv8n | Classify | GPU | FP16 | ? | 1.8ms | 1.7ms | ? | 1.0ms
| YOLOv8n | Classify | CPU | INT8 | × | 3.5ms | × | ? | ×
| YOLOv8n | Classify | GPU | INT8 | × | 8.0ms | × | ? | 1.0ms
| YOLOv8n | Detect | CPU | FP32 | 33.3ms | 33.1ms | 42.2ms | 28.6ms | ×
| YOLOv8n | Detect | GPU | FP32 | 6.4ms | 7.0ms | 6.8ms | ? | 6.0ms
| YOLOv8n | Detect | CPU | FP16 | × | 58.5ms | 41.9ms | 28.6ms | ×
| YOLOv8n | Detect | GPU | FP16 | ? | 19.4ms | 5.7ms | ? | 3.7ms
| YOLOv8n | Detect | CPU | INT8 | × | 38.5ms | × | 24.5ms | ×
| YOLOv8n | Detect | GPU | INT8 | × | 82.5ms | × | ? | 4.7ms
| YOLOv8n | Segment | CPU | FP32 | × | 43.6ms | 54.7ms | 37.5ms | ×
| YOLOv8n | Segment | GPU | FP32 | 9.5ms | 10.6ms | × | ? | 8.1ms
| YOLOv8n | Segment | CPU | FP16 | × | 74.4ms | 54.9ms | 37.4ms | ×
| YOLOv8n | Segment | GPU | FP16 | ? | 27.7ms | × | ? | 5.9ms
| YOLOv8n | Segment | CPU | INT8 | × | 51.4ms | × | ? | ×
| YOLOv8n | Segment | GPU | INT8 | × | 99.9ms | × | ? | 5.6ms
| YOLOv11n | Classify | CPU | FP32 | 4.1ms | 2.6ms | 4.4ms | 2.6ms | ×
| YOLOv11n | Classify | GPU | FP32 | 2.7ms | 1.7ms | × | ? | 1.4ms
| YOLOv11n | Classify | CPU | FP16 | × | 6.6ms | 4.5ms | 2.6ms | ×
| YOLOv11n | Classify | GPU | FP16 | ? | 2.1ms | × | ? | 1.1ms
| YOLOv11n | Classify | CPU | INT8 | × | ? | × | ? | ×
| YOLOv11n | Classify | GPU | INT8 | × | ? | × | ? | 1.3ms
| YOLOv11n | Detect | CPU | FP32 | 35.0ms | 32.1ms | 44.4ms | 25.0ms | ×
| YOLOv11n | Detect | GPU | FP32 | 7.2ms | 7.2ms | × | ? | 6.0ms
| YOLOv11n | Detect | CPU | FP16 | × | 63.8ms | 44.8ms | 25.0ms | ×
| YOLOv11n | Detect | GPU | FP16 | ? | 19.9ms | × | ? | 3.9ms
| YOLOv11n | Detect | CPU | INT8 | × | ? | × | 22.8ms | ×
| YOLOv11n | Detect | GPU | INT8 | × | ? | × | ? | 4.7ms
| YOLOv11n | Segment | CPU | FP32 | × | 43.0ms | 56.9ms | 34.0ms | ×
| YOLOv11n | Segment | GPU | FP32 | x | 10.8ms | × | ? | 7.5ms
| YOLOv11n | Segment | CPU | FP16 | × | 80.4ms | 58.1ms | 33.8ms | ×
| YOLOv11n | Segment | GPU | FP16 | x | 28.1ms | × | ? | 6.2ms
| YOLOv11n | Segment | CPU | INT8 | × | ? | × | ? | ×
| YOLOv11n | Segment | GPU | INT8 | × | ? | × | ? | 4.9ms

You can test Python code with:
```powershell
# Windows 
pip install -r requirements.txt
./run.bat
```
or
```bash
# Linux
pip install -r requirements.txt
./run.sh
```

Python test Ubuntu22.04 in Docker(CPU i7-12700, GPU RTX3070): 
|       Model       |       Task       |       Device       |       Precision       | PyTorch | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :-----: | :---------: | :----: | :------: | :------: |
| YOLOv5n | Classify | CPU | FP32 | 26.3ms | 21.4ms | 33.2ms | 21.8ms | ×
| YOLOv5n | Classify | GPU | FP32 | 15.6ms | 16.1ms | 16.6ms | ? | 17.0ms
| YOLOv5n | Classify | CPU | FP16 | x | 30.3ms | 31.5ms | 21.7ms | ×
| YOLOv5n | Classify | GPU | FP16 | 14.5ms | 18.6ms | 17.4ms | ? | 19.8ms
| YOLOv5n | Classify | CPU | INT8 | x | 28.9ms | × | ? | ×
| YOLOv5n | Classify | GPU | INT8 | x | 54.8ms | × | ? | 18.9ms
| YOLOv5n | Detect | CPU | FP32 | 30.6ms | 27.0ms | 60.0ms | 24.8ms | ×
| YOLOv5n | Detect | GPU | FP32 | 10.4ms | 14.9ms | 10.7ms | ? | 14.3ms
| YOLOv5n | Detect | CPU | FP16 | x | 40.7ms | 59.8ms | 24.8ms | ×
| YOLOv5n | Detect | GPU | FP16 | 12.3ms | 19.6ms | 10.3ms | 30ms | ? | 12.8ms
| YOLOv5n | Detect | CPU | INT8 | x | 33.7ms | × | 23.1ms | ×
| YOLOv5n | Detect | GPU | INT8 | x | 72.9ms | × | ? | 13.8ms
| YOLOv5n | Segment | CPU | FP32 | 159.2ms | 116.1ms | 147.2ms | 47.8ms | ×
| YOLOv5n | Segment | GPU | FP32 | 34.6ms | 49.1ms | 38.0ms | ? | 70.7ms
| YOLOv5n | Segment | CPU | FP16 | x | 138.8ms | 142.2ms | 48.2ms | ×
| YOLOv5n | Segment | GPU | FP16 | 50.9ms | 78.9ms | 52.4ms | ? | 72.6ms
| YOLOv5n | Segment | CPU | INT8 | x | 127.6ms | × | ? | ×
| YOLOv5n | Segment | GPU | INT8 | x | 191.8ms | × | ? | 13.3ms
| YOLOv8n | Classify | CPU | FP32 | 3.5ms | 2.2ms | 4.1ms | 2.3ms | ×
| YOLOv8n | Classify | GPU | FP32 | 2.5ms | 1.6ms | 1.8ms | ? | 3.5ms
| YOLOv8n | Classify | CPU | FP16 | x | 6.3ms | 4.1s | 2.3ms | ×
| YOLOv8n | Classify | GPU | FP16 | ? | 1.7ms | 1.7ms | ? | 2.8ms
| YOLOv8n | Classify | CPU | INT8 | x | 3.7ms | × | ? | ×
| YOLOv8n | Classify | GPU | INT8 | x | 8.2ms | × | ? | 3.0ms
| YOLOv8n | Detect | CPU | FP32 | 59.2ms | 57.8ms | 60.3s | 49.4ms | ×
| YOLOv8n | Detect | GPU | FP32 | 35.5ms | 40.5ms | 29.4ms | ? | 39.1ms
| YOLOv8n | Detect | CPU | FP16 | x | 77.1ms | 61.3ms | 49.6ms | ×
| YOLOv8n | Detect | GPU | FP16 | ? | 60.4ms | 30.8ms | 30ms | ? | 38.1ms
| YOLOv8n | Detect | CPU | INT8 | x | 64.1ms | × | 44.1ms | ×
| YOLOv8n | Detect | GPU | INT8 | x | 138.7ms | × | ? | 40.9ms
| YOLOv8n | Segment | CPU | FP32 | 184.7ms | 157.8ms | 142.3ms | 100.0ms | ×
| YOLOv8n | Segment | GPU | FP32 | 94.3ms | 104.2ms | 88.5ms | ? | 116.6ms
| YOLOv8n | Segment | CPU | FP16 | x | 180.4ms | 144.8s | 99.3ms | ×
| YOLOv8n | Segment | GPU | FP16 | ? | 122.2ms | 108.7ms | ? | 118.7ms
| YOLOv8n | Segment | CPU | INT8 | x | 166.4ms | × | ? | ×
| YOLOv8n | Segment | GPU | INT8 | x | 275.3ms | × | ? | 40.9ms
| YOLOv11n | Classify | CPU | FP32 | 4.1ms | 2.3ms | 4.6ms | 2.5ms | ×
| YOLOv11n | Classify | GPU | FP32 | 2.8ms | 1.7ms | x | ? | 3.7ms
| YOLOv11n | Classify | CPU | FP16 | x | 6.1ms | 4.5ms | 2.5ms | ×
| YOLOv11n | Classify | GPU | FP16 | ? | 1.9ms | x | ? | 3.3ms
| YOLOv11n | Classify | CPU | INT8 | x | ? | × | ? | ×
| YOLOv11n | Classify | GPU | INT8 | x | ? | × | ? | 3.6ms
| YOLOv11n | Detect | CPU | FP32 | 62.2ms | 52.9ms | 66.2ms | 45.2ms | ×
| YOLOv11n | Detect | GPU | FP32 | 38.7ms | 41.2ms | x | ? | 36.6ms
| YOLOv11n | Detect | CPU | FP16 | x | 82.5ms | 63.0ms | 45.1ms | ×
| YOLOv11n | Detect | GPU | FP16 | ? | 58.2ms | x | ? | 38.2ms
| YOLOv11n | Detect | CPU | INT8 | x | ? | × | 50.0ms | ×
| YOLOv11n | Detect | GPU | INT8 | x | ? | × | ? | 39.1ms
| YOLOv11n | Segment | CPU | FP32 | 183.5ms | 152.7ms | 144.1ms | 91.9ms | ×
| YOLOv11n | Segment | GPU | FP32 | 98.2ms | 116.2ms | x | ? | 114.9ms
| YOLOv11n | Segment | CPU | FP16 | x | 185.4ms | 155.2ms | 92.3ms | ×
| YOLOv11n | Segment | GPU | FP16 | ?ms | 130.4ms | x | ? | 120.2ms
| YOLOv11n | Segment | CPU | INT8 | x | ? | × | ? | ×
| YOLOv11n | Segment | GPU | INT8 | x | ? | × | ? | 39.0ms

You Can download some model weights in:  <https://pan.baidu.com/s/19Ua857QSXEQG7k8FV7YKSQ?pwd=syjb>

You can get a docker image with:
```bash
docker pull taify/yolo_inference:latest
```
