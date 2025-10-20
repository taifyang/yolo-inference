# yolo-inference
C++ and Python implementations of YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv2, YOLOv13 inference.

Supported inference backends include Libtorch/PyTorch, ONNXRuntime, OpenCV, OpenVINO, TensorRT. 

Supported task types include Classify, Detect, Segment.

Supported model types include FP32, FP16, INT8.

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

C++ test in Docker with Intel(R) Xeon(R) Gold 5317 CPU , RTX4090 GPU:
|       Model       |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime | OpenCV  | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :------: | :---------: | :-----: | :------: | :------: |
| YOLOv3u           | Detect           | CPU                | FP32                  | 168.2ms  | 136.3ms     | 426.9ms | 207.6ms  | ×        |
| YOLOv3u           | Detect           | GPU                | FP32                  | 10.0ms   | 12.9ms      | 11.8ms  | ?        | 6.4ms    |
| YOLOv3u           | Detect           | CPU                | FP16                  | ×        | 238.5ms     | 429.9ms | 204.6ms  | ×        |
| YOLOv3u           | Detect           | GPU                | FP16                  | 8.1ms    | 20.4ms      | 9.1ms   | ?        | 2.2ms    |
| YOLOv3u           | Detect           | CPU                | INT8                  | ×        | 251.8ms     | ×       | 58.8ms   | ×        |
| YOLOv3u           | Detect           | GPU                | INT8                  | ×        | 242.1ms     | ×       | ?        | 1.5ms    |
| YOLOv4            | Detect           | CPU                | FP32                  | ?        | 189.5ms     | 260.8ms | 105.2ms  | ×        |
| YOLOv4            | Detect           | GPU                | FP32                  | ?        | 13.2ms      | 16.9ms  | ?        | 4.5ms    |
| YOLOv4            | Detect           | CPU                | FP16                  | ×        | 258.6ms     | 259.3ms | 102.8ms  | ×        |
| YOLOv4            | Detect           | GPU                | FP16                  | ?        | 21.7ms      | 14.3ms  | ?        | 2.0ms    |
| YOLOv4            | Detect           | CPU                | INT8                  | ×        | 262.4ms     | ×       | 38.0ms   | ×        |
| YOLOv4            | Detect           | GPU                | INT8                  | ×        | 180.4ms     | ×       | ?        | 1.6ms    |
| YOLOv5n           | Classify         | CPU                | FP32                  | 12.9ms   | 15.6ms      | 21.7ms  | 7.4ms    | ×        |
| YOLOv5n           | Classify         | GPU                | FP32                  | 5.0ms    | 9.6ms       | 6.4ms   | ?        | 3.4ms    |
| YOLOv5n           | Classify         | CPU                | FP16                  | ×        | 19.1ms      | 22.0ms  | 7.5ms    | ×        |
| YOLOv5n           | Classify         | GPU                | FP16                  | 7.8ms    | 12.1ms      | 5.4ms   | ?        | 3.2ms    |
| YOLOv5n           | Classify         | CPU                | INT8                  | ×        | 23.1ms      | ×       | 6.7ms    | ×        |
| YOLOv5n           | Classify         | GPU                | INT8                  | ×        | 22.1ms      | ×       | ?        | 3.2ms    |
| YOLOv5n           | Detect           | CPU                | FP32                  | 20.7ms   | 24.1ms      | 90.0ms  | 8.0ms    | ×        |
| YOLOv5n           | Detect           | GPU                | FP32                  | 5.9ms    | 9.7ms       | 10.2ms  | ?        | 0.9ms    |
| YOLOv5n           | Detect           | CPU                | FP16                  | ×        | 37.3ms      | 90.1ms  | 7.7ms    | ×        |
| YOLOv5n           | Detect           | GPU                | FP16                  | 5.4ms    | 17.4ms      | 8.8ms   | ?        | 0.7ms    |
| YOLOv5n           | Detect           | CPU                | INT8                  | ×        | 32.1ms      | ×       | 5.5ms    | ×        |
| YOLOv5n           | Detect           | GPU                | INT8                  | ×        | 29.1ms      | ×       | ?        | 0.7ms    |
| YOLOv5n           | Segment          | CPU                | FP32                  | 26.3ms   | 30.4ms      | 119.7ms | 12.1ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP32                  | 8.7ms    | 12.6ms      | 12.5ms  | ?        | 2.1ms    |
| YOLOv5n           | Segment          | CPU                | FP16                  | ×        | 49.0ms      | 119.7ms | 12.1ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP16                  | 8.6ms    | 22.7ms      | 11.2ms  | ?        | 1.9ms    |
| YOLOv5n           | Segment          | CPU                | INT8                  | ×        | 43.7ms      | ×       | 10.4ms   | ×        |
| YOLOv5n           | Segment          | GPU                | INT8                  | ×        | 36.5ms      | ×       | ?        | 1.6ms    |
| YOLOv6n           | Detect           | CPU                | FP32                  | 21.0ms   | 15.2ms      | 33.1ms  | 11.0ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP32                  | 5.4ms    | 9.9ms       | 7.0ms   | ?        | 1.1ms    |
| YOLOv6n           | Detect           | CPU                | FP16                  | ×        | 39.8ms      | 32.8ms  | 10.9ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP16                  | 5.2ms    | 19.2ms      | 6.8ms   | ?        | 0.7ms    |
| YOLOv6n           | Detect           | CPU                | INT8                  | ×        | 45.3ms      | ×       | 5.6ms    | ×        |
| YOLOv6n           | Detect           | GPU                | INT8                  | ×        | 41.6ms      | ×       | ?        | 0.7ms    |
| YOLOv7t           | Detect           | CPU                | FP32                  | 29.4ms   | 15.0ms      | 88.5ms  | 13.6ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP32                  | 6.7ms    | 9.8ms       | 8.8ms   | ?        | 1.2ms    |
| YOLOv7t           | Detect           | CPU                | FP16                  | ×        | 47.8ms      | 88.7ms  | 13.4ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP16                  | 6.1ms    | 19.6ms      | 7.8ms   | ?        | 0.8ms    |
| YOLOv7t           | Detect           | CPU                | INT8                  | ×        | 42.3ms      | ×       | 6.6ms    | ×        |
| YOLOv7t           | Detect           | GPU                | INT8                  | ×        | 39.4ms      | ×       | ?        | 0.7ms    |
| YOLOv8n           | Classify         | CPU                | FP32                  | 4.6ms    | 3.4ms       | 5.6ms   | 1.0ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP32                  | 1.2ms    | 1.0ms       | 2.1ms   | ?        | 0.5ms    |
| YOLOv8n           | Classify         | CPU                | FP16                  | ×        | 4.3ms       | 5.6ms   | 1.0ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP16                  | 1.3ms    | 1.3ms       | 2.1ms   | ?        | 0.5ms    |
| YOLOv8n           | Classify         | CPU                | INT8                  | ×        | 4.4ms       | ×       | 0.7ms    | ×        |
| YOLOv8n           | Classify         | GPU                | INT8                  | ×        | 4.2ms       | ×       | ?        | 0.4ms    |
| YOLOv8n           | Detect           | CPU                | FP32                  | 22.8ms   | 28.8ms      | 50.3ms  | 10.8ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP32                  | 5.5ms    | 9.9ms       | 8.0ms   | ?        | 1.1ms    |
| YOLOv8n           | Detect           | CPU                | FP16                  | ×        | 46.0ms      | 50.1ms  | 10.6ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP16                  | 5.3ms    | 19.3ms      | 7.5ms   | ?        | 0.8ms    |
| YOLOv8n           | Detect           | CPU                | INT8                  | ×        | 40.5ms      | ×       | 6.5ms    | ×        |
| YOLOv8n           | Detect           | GPU                | INT8                  | ×        | 37.9ms      | ×       | ?        | 0.7ms    |
| YOLOv8n           | Segment          | CPU                | FP32                  | 33.3ms   | 37.6ms      | 66.0ms  | 15.3ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP32                  | 9.0ms    | 12.4ms      | 11.3ms  | ?        | 2.1ms    |
| YOLOv8n           | Segment          | CPU                | FP16                  | ×        | 59.6ms      | 67.6ms  | 14.9ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP16                  | 8.6ms    | 23.9ms      | 11.0ms  | ?        | 1.7ms    |
| YOLOv8n           | Segment          | CPU                | INT8                  | ×        | 55.2ms      | ×       | 12.2ms   | ×        |
| YOLOv8n           | Segment          | GPU                | INT8                  | ×        | 47.4ms      | ×       | ?        | 1.5ms    |
| YOLOv9t           | Detect           | CPU                | FP32                  | 38.7ms   | 42.3ms      | 67.9ms  | 12.6ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP32                  | 9.6ms    | 12.4ms      | 15.9ms  | ?        | 1.8ms    |
| YOLOv9t           | Detect           | CPU                | FP16                  | ×        | 54.5ms      | 68.3ms  | 12.4ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP16                  | 9.3ms    | 19.3ms      | 14.9ms  | ?        | 1.3ms    |
| YOLOv9t           | Detect           | CPU                | INT8                  | ×        | 60.7ms      | ×       | 8.7ms    | ×        |
| YOLOv9t           | Detect           | GPU                | INT8                  | ×        | 54.1ms      | ×       | ?        | 1.4ms    |
| YOLOv10n          | Detect           | CPU                | FP32                  | 24.1ms   | 30.3ms      | 55.4ms  | 9.8ms    | ×        |
| YOLOv10n          | Detect           | GPU                | FP32                  | 5.7ms    | 11.3ms      | ×       | ?        | 1.2ms    |
| YOLOv10n          | Detect           | CPU                | FP16                  | ×        | 57.9ms      | 54.2ms  | 9.8ms    | ×        |
| YOLOv10n          | Detect           | GPU                | FP16                  | 5.5ms    | 19.0ms      | ×       | ?        | 0.8ms    |
| YOLOv10n          | Detect           | CPU                | INT8                  | ×        | 48.7ms      | ×       | 6.6ms    | ×        |
| YOLOv10n          | Detect           | GPU                | INT8                  | ×        | 44.2ms      | ×       | ?        | 0.8ms    |
| YOLOv11n          | Classify         | CPU                | FP32                  | 5.3ms    | 3.3ms       | 6.6ms   | 1.2ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP32                  | 1.7ms    | 1.2ms       | ×       | ?        | 0.6ms    |
| YOLOv11n          | Classify         | CPU                | FP16                  | ×        | 4.5ms       | 6.4ms   | 1.2ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP16                  | 1.8ms    | 1.5ms       | ×       | ?        | 0.5ms    |
| YOLOv11n          | Classify         | CPU                | INT8                  | ×        | 4.5ms       | ×       | 0.9ms    | ×        |
| YOLOv11n          | Classify         | GPU                | INT8                  | ×        | 4.9ms       | ×       | ?        | 0.5ms    |
| YOLOv11n          | Detect           | CPU                | FP32                  | 26.9ms   | 30.5ms      | 55.7ms  | 10.1ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP32                  | 6.3ms    | 10.3ms      | ×       | ?        | 1.2ms    |
| YOLOv11n          | Detect           | CPU                | FP16                  | ×        | 58.1ms      | 55.5ms  | 10.0ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP16                  | 5.9ms    | 20.0ms      | ×       | ?        | 0.9ms    |
| YOLOv11n          | Detect           | CPU                | INT8                  | ×        | 44.8ms      | ×       | 6.6ms    | ×        |
| YOLOv11n          | Detect           | GPU                | INT8                  | ×        | 39.3ms      | ×       | ?        | 0.8ms    |
| YOLOv11n          | Segment          | CPU                | FP32                  | 34.5ms   | 38.8ms      | 72.2ms  | 14.6ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP32                  | 9.6ms    | 12.4ms      | ×       | ?        | 2.2ms    |
| YOLOv11n          | Segment          | CPU                | FP16                  | ×        | 71.3ms      | 73.5ms  | 14.3ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP16                  | 9.3ms    | 20.9ms      | ×       | ?        | 1.8ms    |
| YOLOv11n          | Segment          | CPU                | INT8                  | ×        | 58.4ms      | ×       | 11.1ms   | ×        |
| YOLOv11n          | Segment          | GPU                | INT8                  | ×        | 50.0ms      | ×       | ?        | 1.7ms    |
| YOLOv12n          | Classify         | CPU                | FP32                  | 8.5ms    | 6.1ms       | 13.2ms  | 3.1ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP32                  | 3.1ms    | 1.8ms       | 5.6ms   | ?        | 0.9ms    |
| YOLOv12n          | Classify         | CPU                | FP16                  | ×        | 8.4ms       | 13.3ms  | 3.1ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP16                  | 3.3ms    | 2.2ms       | 4.8ms   | ?        | 0.7ms    |
| YOLOv12n          | Classify         | CPU                | INT8                  | ×        | 8.2ms       | ×       | 2.9ms    | ×        |
| YOLOv12n          | Classify         | GPU                | INT8                  | ×        | 7.7ms       | ×       | ?        | 0.9ms    |
| YOLOv12n          | Detect           | CPU                | FP32                  | 37.8ms   | 40.0ms      | 146.6ms | 11.9ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP32                  | 8.3ms    | 11.6ms      | 14.6ms  | ?        | 1.6ms    |
| YOLOv12n          | Detect           | CPU                | FP16                  | ×        | 73.8ms      | 145.4ms | 11.7ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP16                  | 8.2ms    | 21.4ms      | 11.2ms  | ?        | 1.1ms    |
| YOLOv12n          | Detect           | CPU                | INT8                  | ×        | 60.4ms      | ×       | 9.2ms    | ×        |
| YOLOv12n          | Detect           | GPU                | INT8                  | ×        | 48.1ms      | ×       | ?        | 1.4ms    |
| YOLOv12n          | Segment          | CPU                | FP32                  | 44.6ms   | 47.9ms      | 167.0ms | 16.5ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP32                  | 11.8ms   | 10.4ms      | 18.0ms  | ?        | 2.5ms    |
| YOLOv12n          | Segment          | CPU                | FP16                  | ×        | 86.8ms      | 165.3ms | 16.3ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP16                  | 11.7ms   | 24.8ms      | 14.8ms  | ?        | 2.0ms    |
| YOLOv12n          | Segment          | CPU                | INT8                  | ×        | 72.9ms      | ×       | 12.8ms   | ×        |
| YOLOv12n          | Segment          | GPU                | INT8                  | ×        | 60.1ms      | ×       | ?        | 2.3ms    |
| YOLOv13n          | Detect           | CPU                | FP32                  | 58.2ms   | 44.3ms      | 164.5ms | 13.8ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP32                  | 9.9ms    | 12.7ms      | 20.8ms  | ?        | 1.8ms    |
| YOLOv13n          | Detect           | CPU                | FP16                  | ×        | 100.2ms     | 165.9ms | 13.7ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP16                  | 9.9ms    | 22.0ms      | 14.6ms  | ?        | 1.4ms    |
| YOLOv13n          | Detect           | CPU                | INT8                  | ×        | 85.6ms      | ×       | 10.8ms   | ×        |
| YOLOv13n          | Detect           | GPU                | INT8                  | ×        | 74.9ms      | ×       | ?        | 2.2ms    |


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

Python test in Docker with Intel(R) Xeon(R) Gold 5317 CPU , RTX4090 GPU
|       Model       |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime | OpenCV  | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :------: | :---------: | :-----: | :------: | :------: |
| YOLOv3u           | Detect           | CPU                | FP32                  | 194.7ms  | 167.6ms     | 466.5ms | 220.0ms  | ×        |
| YOLOv3u           | Detect           | GPU                | FP32                  | 36.3ms   | 42.0ms      | 38.1ms  | ?        | 37.7ms   |
| YOLOv3u           | Detect           | CPU                | FP16                  | ×        | 264.6ms     | 463.4ms | 214.8ms  | ×        |
| YOLOv3u           | Detect           | GPU                | FP16                  | 34.4ms   | 47.0ms      | 34.9ms  | ?        | 34.6ms   |
| YOLOv3u           | Detect           | CPU                | INT8                  | ×        | 286.0ms     | ×       | 81.1ms   | ×        |
| YOLOv3u           | Detect           | GPU                | INT8                  | ×        | 276.6ms     | ×       | ?        | 33.6ms   |
| YOLOv4            | Detect           | CPU                | FP32                  | ?        | 264.7ms     | 347.1ms | 177.9ms  | ×        | 
| YOLOv4            | Detect           | GPU                | FP32                  | ?        | 91.1ms      | 91.7ms  | ?        | 85.3ms   | 
| YOLOv4            | Detect           | CPU                | FP16                  | ×        | 333.0ms     | 346.7ms | 175.9ms  | ×        | 
| YOLOv4            | Detect           | GPU                | FP16                  | ?        | 96.1ms      | 88.0ms  | ?        | 84.4ms   | 
| YOLOv4            | Detect           | CPU                | INT8                  | ×        | 336.6ms     | ×       | 125.3ms  | ×        |
| YOLOv4            | Detect           | GPU                | INT8                  | ×        | 256.6ms     | ×       | ?        | 84.5ms   |
| YOLOv5n           | Classify         | CPU                | FP32                  | 18.5ms   | 23.8ms      | 41.4ms  | 23.6ms   | ×        |
| YOLOv5n           | Classify         | GPU                | FP32                  | 14.5ms   | 16.3ms      | 14.9ms  | ?        | 10.9ms   |
| YOLOv5n           | Classify         | CPU                | FP16                  | ×        | 26.2ms      | 41.2ms  | 24.3ms   | ×        |
| YOLOv5n           | Classify         | GPU                | FP16                  | 15.4ms   | 18.5ms      | 14.8ms  | ?        | 14.6ms   |
| YOLOv5n           | Classify         | CPU                | INT8                  | ×        | 30.1ms      | ×       | 19.4ms   | ×        |
| YOLOv5n           | Classify         | GPU                | INT8                  | ×        | 29.7ms      | ×       | ?        | 14.6ms   |
| YOLOv5n           | Detect           | CPU                | FP32                  | 24.6ms   | 27.2ms      | 94.9ms  | 18.6ms   | ×        |
| YOLOv5n           | Detect           | GPU                | FP32                  | 12.2ms   | 12.1ms      | 14.0ms  | ?        | 6.3ms    |
| YOLOv5n           | Detect           | CPU                | FP16                  | ×        | 37.9ms      | 96.0ms  | 19.8ms   | ×        |
| YOLOv5n           | Detect           | GPU                | FP16                  | 12.1ms   | 17.1ms      | 13.0ms  | ?        | 6.1ms    |
| YOLOv5n           | Detect           | CPU                | INT8                  | ×        | 34.8ms      | ×       | 18.3ms   | ×        |
| YOLOv5n           | Detect           | GPU                | INT8                  | ×        | 32.5ms      | ×       | ?        | 8.3ms    |
| YOLOv5n           | Segment          | CPU                | FP32                  | 118.0ms  | 108.0ms     | 152.0ms | 49.0ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP32                  | 38.8ms   | 46.4ms      | 34.1ms  | ?        | 39.3ms   |
| YOLOv5n           | Segment          | CPU                | FP16                  | ×        | 122.9ms     | 156.2ms | 48.2ms   | ×        |
| YOLOv5n           | Segment          | GPU                | FP16                  | 46.8ms   | 74.0ms      | 41.7ms  | ?        | 44.0ms   |
| YOLOv5n           | Segment          | CPU                | INT8                  | ×        | 119.6ms     | ×       | 55.7ms   | ×        |
| YOLOv5n           | Segment          | GPU                | INT8                  | ×        | 107.9ms     | ×       | ?        | 38.1ms   |
| YOLOv6n           | Detect           | CPU                | FP32                  | 52.0ms   | 47.4ms      | 72.7ms  | 55.7ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP32                  | 33.5ms   | 36.0ms      | 32.2ms  | ?        | 29.9ms   |
| YOLOv6n           | Detect           | CPU                | FP16                  | ×        | 64.3ms      | 73.5ms  | 55.4ms   | ×        |
| YOLOv6n           | Detect           | GPU                | FP16                  | 34.1ms   | 42.3ms      | 31.9ms  | ?        | 29.8ms   |
| YOLOv6n           | Detect           | CPU                | INT8                  | ×        | 75.5ms      | ×       | 41.5ms   | ×        |
| YOLOv6n           | Detect           | GPU                | INT8                  | ×        | 68.6ms      | ×       | ?        | 29.9ms   |
| YOLOv7t           | Detect           | CPU                | FP32                  | 33.4ms   | 20.9ms      | 93.5ms  | 20.4ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP32                  | 13.2ms   | 12.2ms      | 12.2ms  | ?        | 8.9ms    |
| YOLOv7t           | Detect           | CPU                | FP16                  | ×        | 47.8ms      | 93.7ms  | 21.3ms   | ×        |
| YOLOv7t           | Detect           | GPU                | FP16                  | 14.7ms   | 19.7ms      | 13.7ms  | ?        | 6.3ms    |
| YOLOv7t           | Detect           | CPU                | INT8                  | ×        | 45.3ms      | ×       | 19.3ms   | ×        |
| YOLOv7t           | Detect           | GPU                | INT8                  | ×        | 42.7ms      | ×       | ?        | 6.1ms    |
| YOLOv8n           | Classify         | CPU                | FP32                  | 4.5ms    | 3.0ms       | 5.7ms   | 1.5ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP32                  | 1.5ms    | 1.2ms       | 2.5ms   | ?        | 0.7ms    |
| YOLOv8n           | Classify         | CPU                | FP16                  | ×        | 4.5ms       | 5.7ms   | 1.5ms    | ×        |
| YOLOv8n           | Classify         | GPU                | FP16                  | 1.6ms    | 1.6ms       | 2.6ms   | ?        | 0.6ms    |
| YOLOv8n           | Classify         | CPU                | INT8                  | ×        | 4.7ms       | ×       | 1.0ms    | ×        |
| YOLOv8n           | Classify         | GPU                | INT8                  | ×        | 4.6ms       | ×       | ?        | 0.6ms    |
| YOLOv8n           | Detect           | CPU                | FP32                  | 52.1ms   | 57.9ms      | 87.9ms  | 55.7ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP32                  | 34.1ms   | 34.1ms      | 35.5ms  | ?        | 30.5ms   |
| YOLOv8n           | Detect           | CPU                | FP16                  | ×        | 72.3ms      | 88.4ms  | 56.5ms   | ×        |
| YOLOv8n           | Detect           | GPU                | FP16                  | 33.9ms   | 42.4ms      | 33.4ms  | ?        | 29.6ms   |
| YOLOv8n           | Detect           | CPU                | INT8                  | ×        | 68.2ms      | ×       | 43.5ms   | ×        |
| YOLOv8n           | Detect           | GPU                | INT8                  | ×        | 63.1ms      | ×       | ?        | 29.3ms   |
| YOLOv8n           | Segment          | CPU                | FP32                  | 143.6ms  | 133.0ms     | 170.8ms | 99.6ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP32                  | 95.1ms   | 95.6ms      | 93.9ms  | ?        | 92.3ms   |
| YOLOv8n           | Segment          | CPU                | FP16                  | ×        | 153.9ms     | 172.0ms | 98.3ms   | ×        |
| YOLOv8n           | Segment          | GPU                | FP16                  | 95.5ms   | 104.2ms     | 93.5ms  | ?        | 91.8ms   |
| YOLOv8n           | Segment          | CPU                | INT8                  | ×        | 151.1ms     | ×       | 95.0ms   | ×        |
| YOLOv8n           | Segment          | GPU                | INT8                  | ×        | 136.8ms     | ×       | ?        | 85.6ms   |
| YOLOv9t           | Detect           | CPU                | FP32                  | 67.9ms   | 68.4ms      | 104.8ms | 55.8ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP32                  | 37.8ms   | 37.0ms      | 43.1ms  | ?        | 31.0ms   |
| YOLOv9t           | Detect           | CPU                | FP16                  | ×        | 80.9ms      | 103.4ms | 56.9ms   | ×        |
| YOLOv9t           | Detect           | GPU                | FP16                  | 37.8ms   | 43.2ms      | 42.6ms  | ?        | 30.7ms   |
| YOLOv9t           | Detect           | CPU                | INT8                  | ×        | 89.6ms      | ×       | 48.1ms   | ×        |
| YOLOv9t           | Detect           | GPU                | INT8                  | ×        | 80.3ms      | ×       | ?        | 30.6ms   |
| YOLOv10n          | Detect           | CPU                | FP32                  | 53.7ms   | 60.7ms      | 93.7ms  | 54.0ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP32                  | 33.9ms   | 35.5ms      | ×       | ?        | 29.8ms   |
| YOLOv10n          | Detect           | CPU                | FP16                  | ×        | 82.7ms      | 93.1ms  | 53.4ms   | ×        |
| YOLOv10n          | Detect           | GPU                | FP16                  | 34.9ms   | 42.6ms      | ×       | ?        | 29.3ms   |
| YOLOv10n          | Detect           | CPU                | INT8                  | ×        | 79.0ms      | ×       | 45.1ms   | ×        |
| YOLOv10n          | Detect           | GPU                | INT8                  | ×        | 70.4ms      | ×       | ?        | 29.3ms   |
| YOLOv11n          | Classify         | CPU                | FP32                  | 5.2ms    | 3.8ms       | 6.5ms   | 1.7ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP32                  | 2.0ms    | 1.5ms       | ×       | ?        | 0.8ms    |
| YOLOv11n          | Classify         | CPU                | FP16                  | ×        | 5.0ms       | 6.6ms   | 1.7ms    | ×        |
| YOLOv11n          | Classify         | GPU                | FP16                  | 2.1ms    | 1.9ms       | ×       | ?        | 0.7ms    |
| YOLOv11n          | Classify         | CPU                | INT8                  | ×        | 4.9ms       | ×       | 1.1ms    | ×        |
| YOLOv11n          | Classify         | GPU                | INT8                  | ×        | 5.4ms       | ×       | ?        | 0.7ms    |
| YOLOv11n          | Detect           | CPU                | FP32                  | 57.7ms   | 56.6ms      | 94.3ms  | 53.6ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP32                  | 34.8ms   | 35.3ms      | ×       | ?        | 31.6ms   |
| YOLOv11n          | Detect           | CPU                | FP16                  | ×        | 84.5ms      | 93.7ms  | 53.8ms   | ×        |
| YOLOv11n          | Detect           | GPU                | FP16                  | 34.6ms   | 43.8ms      | ×       | ?        | 29.9ms   |
| YOLOv11n          | Detect           | CPU                | INT8                  | ×        | 74.3ms      | ×       | 43.8ms   | ×        |
| YOLOv11n          | Detect           | GPU                | INT8                  | ×        | 64.9ms      | ×       | ?        | 32.4ms   |
| YOLOv11n          | Segment          | CPU                | FP32                  | 143.0ms  | 133.3ms     | 163.4ms | 96.3ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP32                  | 93.2ms   | 91.9ms      | ×       | ?        | 89.8ms   |
| YOLOv11n          | Segment          | CPU                | FP16                  | ×        | 164.1ms     | 160.9ms | 95.4ms   | ×        |
| YOLOv11n          | Segment          | GPU                | FP16                  | 93.3ms   | 102.3ms     | ×       | ?        | 90.2ms   |
| YOLOv11n          | Segment          | CPU                | INT8                  | ×        | 154.4ms     | ×       | 92.4ms   | ×        |
| YOLOv11n          | Segment          | GPU                | INT8                  | ×        | 137.5ms     | ×       | ?        | 88.9ms   |
| YOLOv12n          | Classify         | CPU                | FP32                  | 8.8ms    | 6.1ms       | 13.8ms  | 2.2ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP32                  | 3.5ms    | 2.1ms       | 6.2ms   | ?        | 1.0ms    |
| YOLOv12n          | Classify         | CPU                | FP16                  | ×        | 8.9ms       | 13.7ms  | 2.3ms    | ×        |
| YOLOv12n          | Classify         | GPU                | FP16                  | 3.8ms    | 2.7ms       | 5.4ms   | ?        | 0.9ms    |
| YOLOv12n          | Classify         | CPU                | INT8                  | ×        | 8.7ms       | ×       | 1.9ms    | ×        |
| YOLOv12n          | Classify         | GPU                | INT8                  | ×        | 8.4ms       | ×       | ?        | 1.1ms    |
| YOLOv12n          | Detect           | CPU                | FP32                  | 66.3ms   | 68.6ms      | 192.1ms | 57.1ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP32                  | 37.6ms   | 36.9ms      | 42.2ms  | ?        | 30.0ms   |
| YOLOv12n          | Detect           | CPU                | FP16                  | ×        | 98.5ms      | 190.7ms | 55.4ms   | ×        |
| YOLOv12n          | Detect           | GPU                | FP16                  | 38.0ms   | 44.8ms      | 37.2ms  | ?        | 30.0ms   |
| YOLOv12n          | Detect           | CPU                | INT8                  | ×        | 89.6ms      | ×       | 50.8ms   | ×        |
| YOLOv12n          | Detect           | GPU                | INT8                  | ×        | 74.7ms      | ×       | ?        | 30.1ms   |
| YOLOv12n          | Segment          | CPU                | FP32                  | 152.0ms  | 140.7ms     | 215.0ms | 97.2ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP32                  | 93.7ms   | 93.6ms      | 94.1ms  | ?        | 76.3ms   |
| YOLOv12n          | Segment          | CPU                | FP16                  | ×        | 180.6ms     | 217.0ms | 96.9ms   | ×        |
| YOLOv12n          | Segment          | GPU                | FP16                  | 91.5ms   | 103.0ms     | 94.8ms  | ?        | 89.0ms   |
| YOLOv12n          | Segment          | CPU                | INT8                  | ×        | 165.3ms     | ×       | 97.0ms   | ×        |
| YOLOv12n          | Segment          | GPU                | INT8                  | ×        | 143.9ms     | ×       | ?        | 86.1ms   |
| YOLOv13n          | Detect           | CPU                | FP32                  | 89.6ms   | 73.2ms      | 211.2ms | 56.2ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP32                  | 39.1ms   | 37.5ms      | 48.0ms  | ?        | 31.5ms   |
| YOLOv13n          | Detect           | CPU                | FP16                  | ×        | 124.5ms     | 215.0ms | 56.3ms   | ×        |
| YOLOv13n          | Detect           | GPU                | FP16                  | 39.3ms   | 45.4ms      | 41.2ms  | ?        | 30.7ms   |
| YOLOv13n          | Detect           | CPU                | INT8                  | ×        | 116.7ms     | ×       | 55.0ms   | ×        |
| YOLOv13n          | Detect           | GPU                | INT8                  | ×        | 104.0ms     | ×       | ?        | 31.9ms   |


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

You can download some model weights in: <https://pan.baidu.com/s/1843WW7tNQK1ycqIALje_fA?pwd=adis>
