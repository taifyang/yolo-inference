# yolo-inference
C++ and Python implementations of yolov5 and yolov8 inference.

Supported model backends include libtorch, onnxruntime, opencv, openvino and tensorrt. 

Supported task types include Classify, Detect and Segment.

Supported model types include fp32, fp16 and int8.

You can test C++ code with:
```powershell
# Windows
mkdir build && cd build
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

C++ test on Windows10: (CPU i7-12700, GPU RTX3070, RAM 32GB)
|       Model       |       Task       |       Device       |       Precision       | Libtorch | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :-----------------: | :----------------: | :------------------: | :---------------------: | :------: | :---------: | :----: | :------: | :------: |
| YOLOv5 | Classify | CPU | FP32 | 34ms | 21ms | 29ms | 9ms | ×
| YOLOv5 | Classify | GPU | FP32 | 6ms | 9ms | 7ms | 30ms | 5ms
| YOLOv5 | Classify | CPU | FP16 | × | 22ms | 30ms | 9ms | ×
| YOLOv5 | Classify | GPU | FP16 | 6ms | 10ms | 6ms | 30ms | 4ms
| YOLOv5 | Classify | CPU | INT8 | × | 27ms | × | ? | ×
| YOLOv5 | Classify | GPU | INT8 | × | 33ms | × | ? | 4ms
| YOLOv5 | Detect | CPU | FP32 | 68ms | 26ms | 77ms | 12ms | ×
| YOLOv5 | Detect | GPU | FP32 | 9ms | 10ms | 10ms | 66ms | 5ms
| YOLOv5 | Detect | CPU | FP16 | × | 39ms | 75ms | 11ms | ×
| YOLOv5 | Detect | GPU | FP16 | 8ms | 16ms | 9ms | 66ms | 5ms
| YOLOv5 | Detect | CPU | INT8 | × | 40ms | × | 9ms | ×
| YOLOv5 | Detect | GPU | INT8 | × | 47ms | × | 56ms | 4ms
| YOLOv5 | Segment | CPU | FP32 | × | 33ms | 105ms | 37ms | ×
| YOLOv5 | Segment | GPU | FP32 | 14ms | 16ms | 13ms | 93ms | 8ms
| YOLOv5 | Segment | CPU | FP16 | × | 57ms | 104ms | 19ms | ×
| YOLOv5 | Segment | GPU | FP16 | 13ms | 20ms | 12ms | 93ms | 7ms
| YOLOv5 | Segment | CPU | INT8 | × | 51ms | × | ? | ×
| YOLOv5 | Segment | GPU | INT8 | × | 66ms | × | ? | 6ms
| YOLOv8 | Classify | CPU | FP32 | 13ms | 3ms | 4ms | 3ms | ×
| YOLOv8 | Classify | GPU | FP32 | 2ms | 2ms | × | 7ms | 1ms
| YOLOv8 | Classify | CPU | FP16 | × | 5ms | 4ms | 2ms | ×
| YOLOv8 | Classify | GPU | FP16 | ? | 2ms | × | 16ms | 0.6ms
| YOLOv8 | Classify | CPU | INT8 | × | 5ms | × | ? | ×
| YOLOv8 | Classify | GPU | INT8 | × | 9ms | × | ? | 0.6ms
| YOLOv8 | Detect | CPU | FP32 | 116ms | 36ms | 71ms | 18ms | ×
| YOLOv8 | Detect | GPU | FP32 | 9ms | 11ms | × | 75ms | 7ms
| YOLOv8 | Detect | CPU | FP16 | × | 49ms | 74ms | 18ms | ×
| YOLOv8 | Detect | GPU | FP16 | ? | 13ms | × | 75ms | 5ms
| YOLOv8 | Detect | CPU | INT8 | × | 50ms | × | 12ms | ×
| YOLOv8 | Detect | GPU | INT8 | × | 62ms | × | 58ms | 6ms
| YOLOv8 | Segment | CPU | FP32 | × | 52ms | 99ms | 25ms | ×
| YOLOv8 | Segment | GPU | FP32 | 14ms | 16ms | × | 99ms | 11ms
| YOLOv8 | Segment | CPU | FP16 | × | 68ms | 100ms | 25ms | ×
| YOLOv8 | Segment | GPU | FP16 | × | 18ms | × | 98ms | 8ms
| YOLOv8 | Segment | CPU | INT8 | × | 76ms | × | ? | ×
| YOLOv8 | Segment | GPU | INT8 | × | 81ms | × | ? | 7ms


You can test Python code with:
```powershell
# Windows 
pip install -r requirements.txt
./run.bat
```
or
```bash
# Windows 
pip install -r requirements.txt
./run.sh
```

Python test on Windows10: (CPU i7-12700, GPU RTX3070, RAM 32GB)
|       Model       |       Task       |       Device       |       Precision       | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :-----------------: | :----------------: | :------------------: | :---------------------: | :---------: | :----: | :------: | :------: |
| YOLOv5 | Classify | CPU | FP32 | 21ms | 38ms | 27ms | ×
| YOLOv5 | Classify | GPU | FP32 | 14ms | ? | 36ms | 19ms
| YOLOv5 | Classify | CPU | FP16 | 29ms | 38ms | 27ms | ×
| YOLOv5 | Classify | GPU | FP16 | 15ms | ? | 39ms | 18ms
| YOLOv5 | Classify | CPU | INT8 | 34ms | × | ? | ×
| YOLOv5 | Classify | GPU | INT8 | 38ms | × | ? | 18ms
| YOLOv5 | Detect | CPU | FP32 | 29ms | 103ms | 31ms | ×
| YOLOv5 | Detect | GPU | FP32 | 18ms | ? | 66ms | 17ms
| YOLOv5 | Detect | CPU | FP16 | 42ms | 103ms | 31ms | ×
| YOLOv5 | Detect | GPU | FP16 | 19ms | ? | 63ms | 16ms
| YOLOv5 | Detect | CPU | INT8 | 61ms | × | 38ms | ×
| YOLOv5 | Detect | GPU | INT8 | 53ms | × | 75ms | 16ms
| YOLOv5 | Segment | CPU | FP32 | 39ms | ? | 46ms | ×
| YOLOv5 | Segment | GPU | FP32 | 23ms | ? | 94ms | 20ms
| YOLOv5 | Segment | CPU | FP16 | 57ms | ? | 76ms | ×
| YOLOv5 | Segment | GPU | FP16 | 23ms | ? | 92ms | 19ms
| YOLOv5 | Segment | CPU | INT8 | 63ms | × | ? | ×
| YOLOv5 | Segment | GPU | INT8 | 63ms | × | ? | 17ms
| YOLOv8 | Classify | CPU | FP32 | 3ms | 5ms | 5ms | ×
| YOLOv8 | Classify | GPU | FP32 | 2ms | ? | 17ms | 6ms
| YOLOv8 | Classify | CPU | FP16 | 6ms | 5ms | 4ms | ×
| YOLOv8 | Classify | GPU | FP16 | 2ms | ? | 15ms | 5ms
| YOLOv8 | Classify | CPU | INT8 | 6ms | × | ? | ×
| YOLOv8 | Classify | GPU | INT8 | 7ms | × | ? | 6ms
| YOLOv8 | Detect | CPU | FP32 | 54ms | 88ms | 100ms | ×
| YOLOv8 | Detect | GPU | FP32 | 35ms | ? | 90ms | 37ms
| YOLOv8 | Detect | CPU | FP16 | 75ms | 88ms | 98ms | ×
| YOLOv8 | Detect | GPU | FP16 | 39ms | ? | 91ms | 34ms
| YOLOv8 | Detect | CPU | INT8 | 72ms | × | 50ms | ×
| YOLOv8 | Detect | GPU | INT8 | 87ms | × | 73ms | 36ms
| YOLOv8 | Segment | CPU | FP32 | 74ms | ? | 133ms | ×
| YOLOv8 | Segment | GPU | FP32 | 40ms | ? | 117ms | 41ms
| YOLOv8 | Segment | CPU | FP16 | 94ms | ? | 84ms | ×
| YOLOv8 | Segment | GPU | FP16 | 44ms | ? | 114ms | 38ms
| YOLOv8 | Segment | CPU | INT8 | 92ms | × | ? | ×
| YOLOv8 | Segment | GPU | INT8 | 129ms | × | ? | 37ms