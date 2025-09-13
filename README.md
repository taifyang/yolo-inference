# yolo-inference
C++ and Python implementations of YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv2, YOLOv13 inference.

Supported inference backends include Libtorch/PyTorch, ONNXRuntime, OpenCV, OpenVINO and TensorRT. 

Supported task types include Classify, Detect and Segment.

Supported model types include FP32, FP16 and INT8.

Dependencies(tested):
* [CUDA](https://developer.nvidia.com/cuda-downloads) version 11.8.0/12.5.1/12.8.0
* [OpenCV](https://github.com/opencv/opencv) version 4.9.0/4.10.0/4.11.0 (built with CUDA)
* [ONNXRuntime](https://github.com/microsoft/onnxruntime) version 1.18.1/1.20.0/1.22.0
* [OpenVINO](https://github.com/openvinotoolkit/openvino) version 2024.1.0/2024.4.0/2025.2.0
* [TensorRT](https://developer.nvidia.com/tensorrt/download) version 8.2.1.8/10.6.0.26/10.8.0.43
* [Torch](https://pytorch.org) version 2.0.0+cu118/2.5.0+cu124/2.7.0+cu128

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
mkdir build ; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./run.sh
```

C++ test in Docker(Intel(R) Xeon(R) Gold 5317 CPU , RTX4090 GPU):
|       Model       |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime | OpenCV  | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :------: | :---------: | :-----: | :------: | :------: |
| YOLOv5n           | Classify         | CPU                | FP32                  | 13.8ms   | 19.5ms      | 23.1ms  | 8.6ms    | ×        |
| YOLOv5n           | Classify         | GPU                | FP32                  | 5.3ms    | 9.8ms       | 6.7ms   | ?        | 3.6ms    |
| YOLOv5n           | Classify         | CPU                | FP16                  | ×        | 19.3ms      | 22.7ms  | 8.6ms    | ×        |
| YOLOv5n           | Classify         | GPU                | FP16                  | 8.1ms    | 12.2ms      | 6.0ms   | ?        | 3.5ms    |
| YOLOv5n           | Classify         | CPU                | INT8                  | ×        | 22.9ms      | ×       | 7.8ms    | ×        |
| YOLOv5n           | Classify         | GPU                | INT8                  | ×        | 21.9ms      | ×       | ?        | 3.5ms    |
| YOLOv5n           | Detect           | CPU                | FP32                  | 22.2ms   | 25.2ms      | 91.1ms  | 8.3ms    | ×        |
| YOLOv5n           | Detect           | GPU                | FP32                  | 6.3ms    | 10.1ms      | 10.5ms  | ?        | 1.1ms    |
| YOLOv5n           | Detect           | CPU                | FP16                  | ×        | 37.2ms      | 90.7ms  | 8.2ms    | ×        |
| YOLOv5n           | Detect           | GPU                | FP16                  | 5.8ms    | 17.5ms      | 10.0ms  | ?        | 0.9ms    |
| YOLOv5n           | Detect           | CPU                | INT8                  | ×        | 32.5ms      | ×       | 6.0ms    | ×        |
| YOLOv5n           | Detect           | GPU                | INT8                  | ×        | 29.2ms      | ×       | ?        | 0.8ms    |
| YOLOv5n           | Segment          | CPU                | FP32                  | 27.9ms   | 28.1ms      | 122.1ms | 15.3ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP32                  | 10.5ms   | 12.1ms      | 14.2ms  | ?        | 3.4ms    |
| YOLOv5n           | Segment          | CPU                | FP16                  | ×        | 49.3ms      | 121.5ms | 15.7ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP16                  | 10.1ms   | 22.1ms      | 14.4ms  | ?        | 3.2ms    |
| YOLOv5n           | Segment          | CPU                | INT8                  | ×        | 41.0ms      | ×       | 14.4ms   | ×        |
| YOLOv5n           | Segment          | GPU                | INT8                  | ×        | 35.9ms      | ×       | ?        | 2.3ms    |
| YOLOv6n           | Detect           | CPU                | FP32                  | ?        | 15.1ms      | 33.9ms  | 11.3ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP32                  | ?        | 10.2ms      | 8.0ms   | ?        | 1.4ms    |
| YOLOv6n           | Detect           | CPU                | FP16                  | ×        | 30.1ms      | 33.6ms  | 10.8ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP16                  | ?        | 14.5ms      | 7.9ms   | ?        | 1.0ms    |
| YOLOv6n           | Detect           | CPU                | INT8                  | ×        | 38.9ms      | ×       | 6.0ms    | ×        |
| YOLOv6n           | Detect           | GPU                | INT8                  | ×        | 40.0ms      | ×       | ?        | 1.0ms    |
| YOLOv7t           | Detect           | CPU                | FP32                  | 29.0ms   | 15.7ms      | 89.7ms  | 14.0ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP32                  | 7.6ms    | 10.2ms      | 9.1ms   | ?        | 1.4ms    |
| YOLOv7t           | Detect           | CPU                | FP16                  | ×        | 47.9ms      | 90.1ms  | 14.4ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP16                  | 7.0ms    | 19.7ms      | 8.2ms   | ?        | 1.0ms    |
| YOLOv7t           | Detect           | CPU                | INT8                  | ×        | 42.3ms      | ×       | 7.1ms    | ×        |
| YOLOv7t           | Detect           | GPU                | INT8                  | ×        | 39.8ms      | ×       | ?        | 0.9ms    |
| YOLOv8n           | Classify         | CPU                | FP32                  | 5.3ms    | 3.2ms       | 5.8ms   | 1.6ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP32                  | 1.9ms    | 1.5ms       | 2.7ms   | ?        | 1.0ms    |
| YOLOv8n           | Classify         | CPU                | FP16                  | ×        | 4.7ms       | 5.9ms   | 1.6ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP16                  | 1.9ms    | 1.9ms       | 2.7ms   | ?        | 0.9ms    |
| YOLOv8n           | Classify         | CPU                | INT8                  | ×        | 4.8ms       | ×       | 1.3ms    | ×        |
| YOLOv8n           | Classify         | GPU                | INT8                  | ×        | 4.7ms       | ×       | ?        | 0.8ms    |
| YOLOv8n           | Detect           | CPU                | FP32                  | 22.6ms   | 29.8ms      | 51.5ms  | 11.2ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP32                  | 6.0ms    | 9.8ms       | 8.6ms   | ?        | 1.3ms    |
| YOLOv8n           | Detect           | CPU                | FP16                  | ×        | 46.7ms      | 51.0ms  | 11.1ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP16                  | 5.6ms    | 19.3ms      | 8.3ms   | ?        | 1.1ms    |
| YOLOv8n           | Detect           | CPU                | INT8                  | ×        | 41.0ms      | ×       | 7.1ms    | ×        |
| YOLOv8n           | Detect           | GPU                | INT8                  | ×        | 37.6ms      | ×       | ?        | 0.9ms    |
| YOLOv8n           | Segment          | CPU                | FP32                  | ×        | 36.0ms      | 69.2ms  | 17.7ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP32                  | 12.7ms   | 14.7ms      | 13.2ms  | ?        | 3.5ms    |
| YOLOv8n           | Segment          | CPU                | FP16                  | ×        | 62.9ms      | 71.3ms  | 16.9ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP16                  | 12.7ms   | 25.8ms      | 14.2ms  | ?        | 3.0ms    |
| YOLOv8n           | Segment          | CPU                | INT8                  | ×        | 55.7ms      | ×       | 15.3ms   | ×        |
| YOLOv8n           | Segment          | GPU                | INT8                  | ×        | 45.2ms      | ×       | ?        | 2.2ms    |
| YOLOv9t           | Detect           | CPU                | FP32                  | 39.7ms   | 41.3ms      | 68.5ms  | 13.0ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP32                  | 10.2ms   | 13.2ms      | 16.3ms  | ?        | 2.0ms    |
| YOLOv9t           | Detect           | CPU                | FP16                  | ×        | 55.1ms      | 68.5ms  | 12.6ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP16                  | 10.0ms   | 19.8ms      | 15.8ms  | ?        | 1.7ms    |
| YOLOv9t           | Detect           | CPU                | INT8                  | ×        | 60.9ms      | ×       | 9.5ms    | ×        |
| YOLOv9t           | Detect           | GPU                | INT8                  | ×        | 52.7ms      | ×       | ?        | 1.7ms    |
| YOLOv10n          | Detect           | CPU                | FP32                  | 25.4ms   | 31.6ms      | 55.8ms  | 10.3ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP32                  | 4.9ms    | 11.4ms      | ×       | ?        | 1.4ms    |
| YOLOv10n          | Detect           | CPU                | FP16                  | ×        | 58.2ms      | 55.2ms  | 10.6ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP16                  | 4.8ms    | 18.4ms      | ×       | ?        | 1.1ms    |
| YOLOv10n          | Detect           | CPU                | INT8                  | ×        | 49.9ms      | ×       | 7.2ms    | ×        |
| YOLOv10n          | Detect           | GPU                | INT8                  | ×        | 45.3ms      | ×       | ?        | 1.0ms    |
| YOLOv11n          | Classify         | CPU                | FP32                  | 5.9ms    | 3.8ms       | 6.9ms   | 1.8ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP32                  | 2.3ms    | 1.8ms       | ×       | ?        | 1.1ms    |
| YOLOv11n          | Classify         | CPU                | FP16                  | ×        | 5.2ms       | 7.1ms   | 1.8ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP16                  | 2.4ms    | 2.1ms       | ×       | ?        | 0.9ms    |
| YOLOv11n          | Classify         | CPU                | INT8                  | ×        | 5.2ms       | ×       | 1.5ms    | ×        |
| YOLOv11n          | Classify         | GPU                | INT8                  | ×        | 5.3ms       | ×       | ?        | 1.0ms    |
| YOLOv11n          | Detect           | CPU                | FP32                  | 27.5ms   | 30.6ms      | 58.0ms  | 10.8ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP32                  | 7.3ms    | 10.7ms      | ×       | ?        | 1.5ms    |
| YOLOv11n          | Detect           | CPU                | FP16                  | ×        | 58.3ms      | 57.7ms  | 10.3ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP16                  | 7.0ms    | 19.7ms      | ×       | ?        | 1.1ms    |
| YOLOv11n          | Detect           | CPU                | INT8                  | ×        | 45.7ms      | ×       | 7.1ms    | ×        |
| YOLOv11n          | Detect           | GPU                | INT8                  | ×        | 39.6ms      | ×       | ?        | 1.0ms    |
| YOLOv11n          | Segment          | CPU                | FP32                  | ×        | 36.3ms      | 76.3ms  | 16.9ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP32                  | ×        | 12.4ms      | ×       | ?        | 3.4ms    |
| YOLOv11n          | Segment          | CPU                | FP16                  | ×        | 71.9ms      | 77.8ms  | 16.7ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP16                  | ×        | 23.9ms      | ×       | ?        | 3.1ms    |
| YOLOv11n          | Segment          | CPU                | INT8                  | ×        | 59.0ms      | ×       | 15.4ms   | ×        |
| YOLOv11n          | Segment          | GPU                | INT8                  | ×        | 48.9ms      | ×       | ?        | 2.3ms    |
| YOLOv12n          | Classify         | CPU                | FP32                  | 9.4ms    | 6.4ms       | 14.0ms  | 3.5ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP32                  | 3.7ms    | 2.5ms       | 6.4ms   | ?        | 1.4ms    |
| YOLOv12n          | Classify         | CPU                | FP16                  | ×        | 9.4ms       | 13.7ms  | 3.8ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP16                  | 3.9ms    | 2.9ms       | 5.6ms   | ?        | 1.3ms    |
| YOLOv12n          | Classify         | CPU                | INT8                  | ×        | 8.5ms       | ×       | 3.4ms    | ×        |
| YOLOv12n          | Classify         | GPU                | INT8                  | ×        | 8.5ms       | ×       | ?        | 1.3ms    |
| YOLOv12n          | Detect           | CPU                | FP32                  | 38.2ms   | 38.7ms      | 147.5ms | 12.1ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP32                  | 9.4ms    | 12.1ms      | 15.3ms  | ?        | 1.8ms    |
| YOLOv12n          | Detect           | CPU                | FP16                  | ×        | 73.1ms      | 146.2ms | 12.2ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP16                  | 9.0ms    | 21.5ms      | 12.2ms  | ?        | 1.4ms    |
| YOLOv12n          | Detect           | CPU                | INT8                  | ×        | 60.9ms      | ×       | 9.6ms    | ×        |
| YOLOv12n          | Detect           | GPU                | INT8                  | ×        | 48.9ms      | ×       | ?        | 1.6ms    |
| YOLOv12n          | Segment          | CPU                | FP32                  | ×        | 44.8ms      | 171.4ms | 18.4ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP32                  | ×        | 14.8ms      | 19.9ms  | ?        | 3.8ms    |
| YOLOv12n          | Segment          | CPU                | FP16                  | ×        | 86.7ms      | 172.5ms | 18.3ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP16                  | ×        | 24.0ms      | 16.8ms  | ?        | 3.3ms    |
| YOLOv12n          | Segment          | CPU                | INT8                  | ×        | 71.5ms      | ×       | 16.3ms   | ×        |
| YOLOv12n          | Segment          | GPU                | INT8                  | ×        | 59.5ms      | ×       | ?        | 2.7ms    |
| YOLOv13n          | Detect           | CPU                | FP32                  | 57.2ms   | 44.2ms      | 166.9ms | 13.9ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP32                  | 11.5ms   | 13.0ms      | 20.8ms  | ?        | 2.1ms    |
| YOLOv13n          | Detect           | CPU                | FP16                  | ×        | 103.4ms     | 170.6ms | 14.4ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP16                  | 11.5ms   | 21.9ms      | 15.5ms  | ?        | 1.6ms    |
| YOLOv13n          | Detect           | CPU                | INT8                  | ×        | 90.7ms      | ×       | 11.2ms   | ×        |
| YOLOv13n          | Detect           | GPU                | INT8                  | ×        | 73.5ms      | ×       | ?        | 2.3ms    |

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

Python test in Docker(Intel(R) Xeon(R) Gold 5317 CPU , RTX4090 GPU):
|       Model       |       Task       |       Device       |       Precision       | PyTorch | ONNXRuntime | OpenCV  | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :-----: | :---------: | :-----: | :------: | :------: |
| YOLOv5n           | Classify         | CPU                | FP32                  | 21.3ms  | 21.7ms      | 41.0ms  | 23.3ms   | ×        |
| YOLOv5n           | Classify         | GPU                | FP32                  | 11.6ms  | 15.8ms      | 14.8ms  | ?        | 10.9ms   |
| YOLOv5n           | Classify         | CPU                | FP16                  | ×       | 26.4ms      | 42.2ms  | 24.0ms   | ×        |
| YOLOv5n           | Classify         | GPU                | FP16                  | 15.0ms  | 18.5ms      | 14.2ms  | ?        | 14.2ms   |
| YOLOv5n           | Classify         | CPU                | INT8                  | ×       | 30.8ms      | ×       | 19.5ms   | ×        |
| YOLOv5n           | Classify         | GPU                | INT8                  | ×       | 29.3ms      | ×       | ?        | 14.0ms   |
| YOLOv5n           | Detect           | CPU                | FP32                  | 24.7ms  | 27.2ms      | 94.0ms  | 18.3ms   | ×        |
| YOLOv5n           | Detect           | GPU                | FP32                  | 12.1ms  | 12.0ms      | 13.5ms  | ?        | 6.3ms    |
| YOLOv5n           | Detect           | CPU                | FP16                  | ×       | 37.6ms      | 94.1ms  | 19.9ms   | ×        |
| YOLOv5n           | Detect           | GPU                | FP16                  | 12.1ms  | 17.1ms      | 13.1ms  | ?        | 6.1ms    |
| YOLOv5n           | Detect           | CPU                | INT8                  | ×       | 35.0ms      | ×       | 18.3ms   | ×        |
| YOLOv5n           | Detect           | GPU                | INT8                  | ×       | 32.3ms      | ×       | ?        | 7.8ms    |
| YOLOv5n           | Segment          | CPU                | FP32                  | 117.4ms | 108.0ms     | 150.9ms | 48.2ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP32                  | 38.3ms  | 44.1ms      | 43.6ms  | ?        | 39.8ms   |
| YOLOv5n           | Segment          | CPU                | FP16                  | ×       | 124.4ms     | 148.4ms | 48.8ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP16                  | 46.9ms  | 74.2ms      | 41.0ms  | ?        | 44.2ms   |
| YOLOv5n           | Segment          | CPU                | INT8                  | ×       | 119.6ms     | ×       | 60.4ms   | ×        |
| YOLOv5n           | Segment          | GPU                | INT8                  | ×       | 108.0ms     | ×       | ?        | 7.4ms    |
| YOLOv6n           | Detect           | CPU                | FP32                  | ×       | 43.8ms      | 74.9ms  | 59.1ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP32                  | ×       | 37.1ms      | 35.4ms  | ?        | 33.2ms   |
| YOLOv6n           | Detect           | CPU                | FP16                  | ×       | 56.5ms      | 76.6ms  | 58.4ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP16                  | ×       | 40.1ms      | 35.3ms  | ?        | 32.6ms   |
| YOLOv6n           | Detect           | CPU                | INT8                  | ×       | 68.3ms      | ×       | 43.1ms   | ×        |
| YOLOv6n           | Detect           | GPU                | INT8                  | ×       | 68.6ms      | ×       | ?        | 32.1ms   |
| YOLOv7t           | Detect           | CPU                | FP32                  | 32.7ms  | 21.0ms      | 93.6ms  | 20.5ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP32                  | 9.9ms   | 12.1ms      | 12.2ms  | ?        | 8.9ms    |
| YOLOv7t           | Detect           | CPU                | FP16                  | ×       | 48.0ms      | 94.3ms  | 21.9ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP16                  | 14.6ms  | 19.2ms      | 13.6ms  | ?        | 6.3ms    |
| YOLOv7t           | Detect           | CPU                | INT8                  | ×       | 45.7ms      | ×       | 19.8ms   | ×        |
| YOLOv7t           | Detect           | GPU                | INT8                  | ×       | 42.9ms      | ×       | ?        | 5.9ms    |
| YOLOv8n           | Classify         | CPU                | FP32                  | 4.9ms   | 3.0ms       | 5.7ms   | 1.5ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP32                  | 1.6ms   | 1.2ms       | 2.5ms   | ?        | 0.7ms    |
| YOLOv8n           | Classify         | CPU                | FP16                  | ×       | 4.5ms       | 5.8ms   | 1.5ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP16                  | 1.6ms   | 1.6ms       | 2.5ms   | ?        | 0.6ms    |
| YOLOv8n           | Classify         | CPU                | INT8                  | ×       | 4.7ms       | ×       | 1.0ms    | ×        |
| YOLOv8n           | Classify         | GPU                | INT8                  | ×       | 4.6ms       | ×       | ?        | 0.6ms    |
| YOLOv8n           | Detect           | CPU                | FP32                  | 51.1ms  | 57.9ms      | 90.0ms  | 55.6ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP32                  | 31.1ms  | 34.4ms      | 34.8ms  | ?        | 29.3ms   |
| YOLOv8n           | Detect           | CPU                | FP16                  | ×       | 70.3ms      | 88.4ms  | 56.0ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP16                  | 31.8ms  | 43.0ms      | 34.4ms  | ?        | 29.7ms   |
| YOLOv8n           | Detect           | CPU                | INT8                  | ×       | 70.8ms      | ×       | 43.1ms   | ×        |
| YOLOv8n           | Detect           | GPU                | INT8                  | ×       | 62.4ms      | ×       | ?        | 29.0ms   |
| YOLOv8n           | Segment          | CPU                | FP32                  | 147.2ms | 136.4ms     | 168.3ms | 99.1ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP32                  | 95.6ms  | 94.3ms      | 93.9ms  | ?        | 92.5ms   |
| YOLOv8n           | Segment          | CPU                | FP16                  | ×       | 157.7ms     | 169.9ms | 98.2ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP16                  | 95.8ms  | 103.6ms     | 93.9ms  | ?        | 92.0ms   |
| YOLOv8n           | Segment          | CPU                | INT8                  | ×       | 153.4ms     | ×       | 95.1ms   | ×        |
| YOLOv8n           | Segment          | GPU                | INT8                  | ×       | 139.0ms     | ×       | ?        | 29.9ms   |
| YOLOv9t           | Detect           | CPU                | FP32                  | 65.3ms  | 68.0ms      | 109.6ms | 56.5ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP32                  | 36.1ms  | 37.5ms      | 43.0ms  | ?        | 30.7ms   |
| YOLOv9t           | Detect           | CPU                | FP16                  | ×       | 81.8ms      | 105.7ms | 56.4ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP16                  | 36.4ms  | 42.6ms      | 42.5ms  | ?        | 30.9ms   |
| YOLOv9t           | Detect           | CPU                | INT8                  | ×       | 89.5ms      | ×       | 47.8ms   | ×        |
| YOLOv9t           | Detect           | GPU                | INT8                  | ×       | 80.7ms      | ×       | ?        | 31.2ms   |
| YOLOv10n          | Detect           | CPU                | FP32                  | 26.9ms  | 59.3ms      | 93.9ms  | 53.4ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP32                  | 6.8ms   | 36.0ms      | ×       | ?        | 30.2ms   |
| YOLOv10n          | Detect           | CPU                | FP16                  | ×       | 82.7ms      | 93.9ms  | 53.6ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP16                  | 7.0ms   | 42.9ms      | ×       | ?        | 30.2ms   |
| YOLOv10n          | Detect           | CPU                | INT8                  | ×       | 77.9ms      | ×       | 44.7ms   | ×        |
| YOLOv10n          | Detect           | GPU                | INT8                  | ×       | 70.7ms      | ×       | ?        | 30.1ms   |
| YOLOv11n          | Classify         | CPU                | FP32                  | 5.1ms   | 3.5ms       | 6.6ms   | 1.7ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP32                  | 2.0ms   | 1.5ms       | ×       | ?        | 0.8ms    |
| YOLOv11n          | Classify         | CPU                | FP16                  | ×       | 4.9ms       | 6.6ms   | 1.7ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP16                  | 2.1ms   | 1.9ms       | ×       | ?        | 0.7ms    |
| YOLOv11n          | Classify         | CPU                | INT8                  | ×       | 4.9ms       | ×       | 1.1ms    | ×        |
| YOLOv11n          | Classify         | GPU                | INT8                  | ×       | 5.3ms       | ×       | ?        | 0.7ms    |
| YOLOv11n          | Detect           | CPU                | FP32                  | 57.8ms  | 56.7ms      | 94.9ms  | 53.6ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP32                  | 34.8ms  | 35.6ms      | ×       | ?        | 30.6ms   |
| YOLOv11n          | Detect           | CPU                | FP16                  | ×       | 84.4ms      | 95.6ms  | 53.8ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP16                  | 35.0ms  | 43.6ms      | ×       | ?        | 29.9ms   |
| YOLOv11n          | Detect           | CPU                | INT8                  | ×       | 72.9ms      | ×       | 43.3ms   | ×        |
| YOLOv11n          | Detect           | GPU                | INT8                  | ×       | 65.2ms      | ×       | ?        | 29.2ms   |
| YOLOv11n          | Segment          | CPU                | FP32                  | 145.7ms | 131.6ms     | 169.8ms | 95.7ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP32                  | 94.1ms  | 92.6ms      | ×       | ?        | 87.7ms   |
| YOLOv11n          | Segment          | CPU                | FP16                  | ×       | 163.8ms     | 163.9ms | 95.5ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP16                  | 96.3ms  | 101.3ms     | ×       | ?        | 88.6ms   |
| YOLOv11n          | Segment          | CPU                | INT8                  | ×       | 148.3ms     | ×       | 92.4ms   | ×        |
| YOLOv11n          | Segment          | GPU                | INT8                  | ×       | 137.8ms     | ×       | ?        | 30.4ms   |
| YOLOv12n          | Classify         | CPU                | FP32                  | 8.5ms   | 6.1ms       | 13.2ms  | 2.3ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP32                  | 3.6ms   | 2.1ms       | 6.2ms   | ?        | 1.0ms    |
| YOLOv12n          | Classify         | CPU                | FP16                  | ×       | 8.8ms       | 16.5ms  | 2.3ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP16                  | 3.8ms   | 2.7ms       | 5.3ms   | ?        | 0.9ms    |
| YOLOv12n          | Classify         | CPU                | INT8                  | ×       | 8.8ms       | ×       | 1.9ms    | ×        |
| YOLOv12n          | Classify         | GPU                | INT8                  | ×       | 8.4ms       | ×       | ?        | 1.1ms    |
| YOLOv12n          | Detect           | CPU                | FP32                  | 63.0ms  | 68.6ms      | 191.2ms | 56.6ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP32                  | 36.7ms  | 36.3ms      | 41.9ms  | ?        | 30.9ms   |
| YOLOv12n          | Detect           | CPU                | FP16                  | ×       | 98.1ms      | 189.5ms | 56.3ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP16                  | 38.4ms  | 44.8ms      | 37.1ms  | ?        | 30.8ms   |
| YOLOv12n          | Detect           | CPU                | INT8                  | ×       | 93.8ms      | ×       | 52.2ms   | ×        |
| YOLOv12n          | Detect           | GPU                | INT8                  | ×       | 74.5ms      | ×       | ?        | 30.2ms   |
| YOLOv12n          | Segment          | CPU                | FP32                  | 155.3ms | 141.7ms     | 218.6ms | 97.3ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP32                  | 96.4ms  | 94.0ms      | 96.1ms  | ?        | 73.7ms   |
| YOLOv12n          | Segment          | CPU                | FP16                  | ×       | 179.5ms     | 207.8ms | 97.0ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP16                  | 97.8ms  | 102.8ms     | 93.6ms  | ?        | 89.7ms   |
| YOLOv12n          | Segment          | CPU                | INT8                  | ×       | 165.3ms     | ×       | 95.1ms   | ×        |
| YOLOv12n          | Segment          | GPU                | INT8                  | ×       | 147.5ms     | ×       | ?        | 30.8ms   |
| YOLOv13n          | Detect           | CPU                | FP32                  | 83.6ms  | 75.2ms      | 213.3ms | 57.0ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP32                  | 39.2ms  | 37.0ms      | 47.9ms  | ?        | 31.2ms   |
| YOLOv13n          | Detect           | CPU                | FP16                  | ×       | 124.8ms     | 214.3ms | 57.1ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP16                  | 39.7ms  | 45.2ms      | 41.3ms  | ?        | 30.7ms   |
| YOLOv13n          | Detect           | CPU                | INT8                  | ×       | 116.9ms     | ×       | 54.7ms   | ×        |
| YOLOv13n          | Detect           | GPU                | INT8                  | ×       | 100.5ms     | ×       | ?        | 31.7ms   |


You can get a docker image with:
```bash
docker pull taify/yolo_inference:cuda11.8
```
or
```bash
docker pull taify/yolo_inference:cuda12.5
```
or
```bash
docker pull taify/yolo_inference:cuda12.8
```

You Can download some model weights in:  <https://pan.baidu.com/s/1L8EyTa59qu_eEb3lKRnPQA?pwd=itda>


For your own model, you should transpose output dims for YOLOv8, YOLOv9, YOLOv11, YOLOv12, YOLOv13 detection and segmentation. For onnx model, you can use a scirpt like this:
 ```python
import onnx
import onnx.helper as helper
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage:\n python transpose.py yolov8n.onnx")
        return 1

    file = sys.argv[1]
    if not os.path.exists(file):
        print(f"Not exist path: {file}")
        return 1

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".trans" + suffix

    model = onnx.load(file)
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```