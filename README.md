# yolo-inference
C++ and Python implementations of YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv13 inference.

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

C++ test in Docker with 25 vCPU Intel(R) Xeon(R) Platinum 8470Q , RTX 5090(32GB):
|       Model        |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime  | OpenCV  | OpenVINO | TensorRT |
| :----------------: | :--------------: | :----------------: | :-------------------: | :------: | :----------: | :-----: | :------: | :------: |
| YOLOv3u            | Detect           | CPU                | FP32                  | 167.3ms  | 252.5ms      | 464.0ms | 90.1ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP32                  | 10.0ms   | 11.9ms       | 26.2ms  | ?        | 6.3ms    |
| YOLOv3u            | Detect           | CPU                | FP16                  | ×        | 298.6ms      | 460.2ms | 89.8ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP16                  | 7.7ms    | 18.3ms       | 7.6ms   | ?        | 2.4ms    |
| YOLOv3u            | Detect           | CPU                | INT8                  | ×        | 243.3ms      | ×       | 95.2ms   | ×        |
| YOLOv3u            | Detect           | GPU                | INT8                  | ×        | 237.1ms      | ×       | ?        | 1.7ms    |
| YOLOv4             | Detect           | CPU                | FP32                  | ?        | 285.7ms      | 284.6ms | 64.5ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP32                  | ?        | 11.5ms       | 19.0ms  | ?        | 4.8ms    |
| YOLOv4             | Detect           | CPU                | FP16                  | ×        | 342.3ms      | 281.5ms | 64.2ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP16                  | ?        | 20.4ms       | 11.3ms  | ?        | 2.3ms    |
| YOLOv4             | Detect           | CPU                | INT8                  | ×        | 319.1ms      | ×       | 63.8ms   | ×        |
| YOLOv4             | Detect           | GPU                | INT8                  | ×        | 189.2ms      | ×       | ?        | 1.9ms    |
| YOLOv5n            | Classify         | CPU                | FP32                  | 14.8ms   | 17.2ms       | 24.1ms  | 6.7ms    | ×        |
| YOLOv5n            | Classify         | GPU                | FP32                  | 5.0ms    | 8.5ms        | 4.6ms   | ?        | 3.1ms    |
| YOLOv5n            | Classify         | CPU                | FP16                  | ×        | 18.8ms       | 23.8ms  | 6.6ms    | ×        |
| YOLOv5n            | Classify         | GPU                | FP16                  | 7.0ms    | 11.0ms       | 4.5ms   | ?        | 2.9ms    |
| YOLOv5n            | Classify         | CPU                | INT8                  | ×        | 22.5ms       | ×       | 6.8ms    | ×        |
| YOLOv5n            | Classify         | GPU                | INT8                  | ×        | 21.6ms       | ×       | ?        | 2.9ms    |
| YOLOv5n            | Detect           | CPU                | FP32                  | 22.6ms   | 23.3ms       | 87.7ms  | 11.0ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP32                  | 4.9ms    | 7.7ms        | 7.1ms   | ?        | 1.1ms    |
| YOLOv5n            | Detect           | CPU                | FP16                  | ×        | 35.1ms       | 87.2ms  | 10.8ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP16                  | 4.6ms    | 14.3ms       | 6.5ms   | ?        | 0.9ms    |
| YOLOv5n            | Detect           | CPU                | INT8                  | ×        | 31.4ms       | ×       | 12.3ms   | ×        |
| YOLOv5n            | Detect           | GPU                | INT8                  | ×        | 27.4ms       | ×       | ?        | 0.8ms    |
| YOLOv5n            | Segment          | CPU                | FP32                  | 27.2ms   | 31.4ms       | 116.6ms | 13.3ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP32                  | 7.9ms    | 9.7ms        | 9.2ms   | ?        | 1.6ms    |
| YOLOv5n            | Segment          | CPU                | FP16                  | ×        | 42.8ms       | 116.1ms | 13.1ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP16                  | 7.6ms    | 19.2ms       | 8.7ms   | ?        | 1.3ms    |
| YOLOv5n            | Segment          | CPU                | INT8                  | ×        | 39.7ms       | ×       | 14.6ms   | ×        |
| YOLOv5n            | Segment          | GPU                | INT8                  | ×        | 33.5ms       | ×       | ?        | 1.1ms    |
| YOLOv6n            | Detect           | CPU                | FP32                  | 24.0ms   | 19.8ms       | 36.3ms  | 11.2ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP32                  | 4.6ms    | 8.1ms        | 5.4ms   | ?        | 1.1ms    |
| YOLOv6n            | Detect           | CPU                | FP16                  | ×        | 38.5ms       | 35.9ms  | 11.0ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP16                  | 4.5ms    | 17.0ms       | 6.3ms   | ?        | 0.8ms    |
| YOLOv6n            | Detect           | CPU                | INT8                  | ×        | 47.0ms       | ×       | 10.6ms   | ×        |
| YOLOv6n            | Detect           | GPU                | INT8                  | ×        | 42.9ms       | ×       | ?        | 0.8ms    |
| YOLOv7t            | Detect           | CPU                | FP32                  | 29.8ms   | 19.7ms       | 86.9ms  | 12.4ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP32                  | 5.6ms    | 7.8ms        | 6.0ms   | ?        | 1.3ms    |
| YOLOv7t            | Detect           | CPU                | FP16                  | ×        | 45.2ms       | 86.5ms  | 12.2ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP16                  | 5.1ms    | 16.9ms       | 5.5ms   | ?        | 1.0ms    |
| YOLOv7t            | Detect           | CPU                | INT8                  | ×        | 38.5ms       | ×       | 12.5ms   | ×        |
| YOLOv7t            | Detect           | GPU                | INT8                  | ×        | 39.6ms       | ×       | ?        | 0.9ms    |
| YOLOv8n            | Classify         | CPU                | FP32                  | 5.2ms    | 3.1ms        | 5.0ms   | 3.2ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP32                  | 1.2ms    | 1.1ms        | 1.8ms   | ?        | 0.7ms    |
| YOLOv8n            | Classify         | CPU                | FP16                  | ×        | 4.0ms        | 4.9ms   | 3.1ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP16                  | 1.3ms    | 1.4ms        | 1.9ms   | ?        | 0.6ms    |
| YOLOv8n            | Classify         | CPU                | INT8                  | ×        | 3.7ms        | ×       | 3.6ms    | ×        |
| YOLOv8n            | Classify         | GPU                | INT8                  | ×        | 3.6ms        | ×       | ?        | 0.6ms    |
| YOLOv8n            | Detect           | CPU                | FP32                  | 24.6ms   | 34.4ms       | 64.6ms  | 12.3ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP32                  | 4.7ms    | 8.2ms        | 6.2ms   | ?        | 1.3ms    |
| YOLOv8n            | Detect           | CPU                | FP16                  | ×        | 43.8ms       | 64.2ms  | 12.1ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP16                  | 4.5ms    | 17.1ms       | 6.0ms   | ?        | 0.9ms    |
| YOLOv8n            | Detect           | CPU                | INT8                  | ×        | 43.2ms       | ×       | 15.4ms   | ×        |
| YOLOv8n            | Detect           | GPU                | INT8                  | ×        | 38.1ms       | ×       | ?        | 0.8ms    |
| YOLOv8n            | Segment          | CPU                | FP32                  | 33.0ms   | 39.2ms       | 70.0ms  | 15.3ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP32                  | 7.3ms    | 10.1ms       | 8.3ms   | ?        | 1.9ms    |
| YOLOv8n            | Segment          | CPU                | FP16                  | ×        | 51.5ms       | 69.6ms  | 15.1ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP16                  | 7.5ms    | 21.5ms       | 8.4ms   | ?        | 1.5ms    |
| YOLOv8n            | Segment          | CPU                | INT8                  | ×        | 49.2ms       | ×       | 18.7ms   | ×        |
| YOLOv8n            | Segment          | GPU                | INT8                  | ×        | 42.4ms       | ×       | ?        | 1.4ms    |
| YOLOv9t            | Detect           | CPU                | FP32                  | 43.8ms   | 49.0ms       | 61.0ms  | 17.7ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP32                  | 7.5ms    | 10.3ms       | 11.2ms  | ?        | 2.2ms    |
| YOLOv9t            | Detect           | CPU                | FP16                  | ×        | 52.3ms       | 60.6ms  | 17.5ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP16                  | 7.9ms    | 16.6ms       | 12.8ms  | ?        | 1.7ms    |
| YOLOv9t            | Detect           | CPU                | INT8                  | ×        | 56.9ms       | ×       | 23.8ms   | ×        |
| YOLOv9t            | Detect           | GPU                | INT8                  | ×        | 45.4ms       | ×       | ?        | 1.9ms    |
| YOLOv9c            | Segment          | CPU                | FP32                  | 137.4ms  | 175.1ms      | 308.9ms | 64.2ms   | ×        |
| YOLOv9c            | Segment          | GPU                | FP32                  | 10.9ms   | 27.5ms       | 19.0ms  | ?        | 5.2ms    |
| YOLOv9c            | Segment          | CPU                | FP16                  | ×        | 182.6ms      | 308.5ms | 63.9ms   | ×        |
| YOLOv9c            | Segment          | GPU                | FP16                  | 9.5ms    | 48.0ms       | 12.6ms  | ?        | 2.8ms    |
| YOLOv9c            | Segment          | CPU                | INT8                  | ×        | 178.1ms      | ×       | 67.4ms   | ×        |
| YOLOv9c            | Segment          | GPU                | INT8                  | ×        | 159.0ms      | ×       | ?        | 2.2ms    |
| YOLOv10n           | Detect           | CPU                | FP32                  | 25.9ms   | 30.0ms       | 54.7ms  | 13.0ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP32                  | 4.9ms    | 9.0ms        | ×       | ?        | 1.3ms    |
| YOLOv10n           | Detect           | CPU                | FP16                  | ×        | 55.3ms       | 54.3ms  | 12.8ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP16                  | 4.8ms    | 16.2ms       | ×       | ?        | 1.0ms    |
| YOLOv10n           | Detect           | CPU                | INT8                  | ×        | 50.0ms       | ×       | 15.7ms   | ×        |
| YOLOv10n           | Detect           | GPU                | INT8                  | ×        | 44.2ms       | ×       | ?        | 1.0ms    |
| YOLOv11n           | Classify         | CPU                | FP32                  | 5.7ms    | 3.1ms        | 5.0ms   | 3.4ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP32                  | 1.6ms    | 1.3ms        | ×       | ?        | 0.8ms    |
| YOLOv11n           | Classify         | CPU                | FP16                  | ×        | 4.3ms        | 4.9ms   | 3.3ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP16                  | 1.6ms    | 1.7ms        | ×       | ?        | 0.7ms    |
| YOLOv11n           | Classify         | CPU                | INT8                  | ×        | 4.2ms        | ×       | 4.0ms    | ×        |
| YOLOv11n           | Classify         | GPU                | INT8                  | ×        | 4.4ms        | ×       | ?        | 0.7ms    |
| YOLOv11n           | Detect           | CPU                | FP32                  | 29.6ms   | 30.2ms       | 57.9ms  | 13.5ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP32                  | 5.1ms    | 8.4ms        | ×       | ?        | 1.4ms    |
| YOLOv11n           | Detect           | CPU                | FP16                  | ×        | 55.5ms       | 57.5ms  | 13.3ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP16                  | 5.2ms    | 17.6ms       | ×       | ?        | 1.1ms    |
| YOLOv11n           | Detect           | CPU                | INT8                  | ×        | 42.6ms       | ×       | 15.9ms   | ×        |
| YOLOv11n           | Detect           | GPU                | INT8                  | ×        | 38.1ms       | ×       | ?        | 1.0ms    |
| YOLOv11n           | Segment          | CPU                | FP32                  | 35.8ms   | 39.8ms       | 77.9ms  | 16.4ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP32                  | 8.0ms    | 11.8ms       | ×       | ?        | 2.0ms    |
| YOLOv11n           | Segment          | CPU                | FP16                  | ×        | 68.0ms       | 77.5ms  | 16.1ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP16                  | 8.7ms    | 21.5ms       | ×       | ?        | 1.6ms    |
| YOLOv11n           | Segment          | CPU                | INT8                  | ×        | 55.5ms       | ×       | 20.0ms   | ×        |
| YOLOv11n           | Segment          | GPU                | INT8                  | ×        | 47.9ms       | ×       | ?        | 1.5ms    |
| YOLOv12n           | Classify         | CPU                | FP32                  | 9.4ms    | 4.7ms        | 12.5ms  | 8.6ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP32                  | 2.5ms    | 1.9ms        | 4.5ms   | ?        | 1.0ms    |
| YOLOv12n           | Classify         | CPU                | FP16                  | ×        | 7.9ms        | 12.4ms  | 8.5ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP16                  | 2.6ms    | 2.2ms        | 5.3ms   | ?        | 0.9ms    |
| YOLOv12n           | Classify         | CPU                | INT8                  | ×        | 7.6ms        | ×       | 10.7ms   | ×        |
| YOLOv12n           | Classify         | GPU                | INT8                  | ×        | 7.6ms        | ×       | ?        | 1.0ms    |
| YOLOv12n           | Detect           | CPU                | FP32                  | 39.2ms   | 39.0ms       | 158.0ms | 16.8ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP32                  | 6.6ms    | 9.4ms        | 10.5ms  | ?        | 1.7ms    |
| YOLOv12n           | Detect           | CPU                | FP16                  | ×        | 71.2ms       | 157.6ms | 16.6ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP16                  | 6.7ms    | 19.1ms       | 11.3ms  | ?        | 1.3ms    |
| YOLOv12n           | Detect           | CPU                | INT8                  | ×        | 61.3ms       | ×       | 20.6ms   | ×        |
| YOLOv12n           | Detect           | GPU                | INT8                  | ×        | 52.1ms       | ×       | ?        | 1.7ms    |
| YOLOv12n           | Segment          | CPU                | FP32                  | 47.6ms   | 47.8ms       | 191.6ms | 22.0ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP32                  | 9.6ms    | 12.5ms       | 13.2ms  | ?        | 2.3ms    |
| YOLOv12n           | Segment          | CPU                | FP16                  | ×        | 84.2ms       | 191.2ms | 21.8ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP16                  | 9.6ms    | 22.0ms       | 13.6ms  | ?        | 1.8ms    |
| YOLOv12n           | Segment          | CPU                | INT8                  | ×        | 74.4ms       | ×       | 25.4ms   | ×        |
| YOLOv12n           | Segment          | GPU                | INT8                  | ×        | 57.6ms       | ×       | ?        | 2.2ms    |
| YOLOv13n           | Detect           | CPU                | FP32                  | 50.1ms   | 45.7ms       | 183.1ms | 20.3ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP32                  | 7.9ms    | 9.9ms        | 14.3ms  | ?        | 2.1ms    |
| YOLOv13n           | Detect           | CPU                | FP16                  | ×        | 97.6ms       | 182.7ms | 20.1ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP16                  | 7.9ms    | 19.2ms       | 16.4ms  | ?        | 1.7ms    |
| YOLOv13n           | Detect           | CPU                | INT8                  | ×        | 109.2ms      | ×       | 22.3ms   | ×        |
| YOLOv13n           | Detect           | GPU                | INT8                  | ×        | 84.4ms       | ×       | ?        | 2.6ms    |


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

Python test in Docker with 25 vCPU Intel(R) Xeon(R) Platinum 8470Q , RTX 5090(32GB):
|       Model        |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime  | OpenCV  | OpenVINO | TensorRT |
| :----------------: | :--------------: | :----------------: | :-------------------: | :------: | :----------: | :-----: | :------: | :------: |
| YOLOv3u            | Detect           | CPU                | FP32                  | 170.9ms  | 258.3ms      | 479.1ms | 57.6ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP32                  | 13.4ms   | 16.8ms       | 28.0ms  | ?        | 9.4ms    |
| YOLOv3u            | Detect           | CPU                | FP16                  | ×        | 302.7ms      | 475.3ms | 57.8ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP16                  | 11.4ms   | 22.3ms       | 9.7ms   | ?        | 5.3ms    |
| YOLOv3u            | Detect           | CPU                | INT8                  | ×        | 310.5ms      | ×       | 58.4ms   | ×        |
| YOLOv3u            | Detect           | GPU                | INT8                  | ×        | 244.1ms      | ×       | ?        | 4.5ms    |
| YOLOv4             | Detect           | CPU                | FP32                  | ?        | 278.5ms      | 272.0ms | 52.6ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP32                  | ?        | 19.2ms       | 22.7ms  | ?        | 12.0ms   |
| YOLOv4             | Detect           | CPU                | FP16                  | ×        | 356.4ms      | 268.7ms | 52.8ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP16                  | ?        | 25.7ms       | 16.3ms  | ?        | 9.1ms    |
| YOLOv4             | Detect           | CPU                | INT8                  | ×        | 319.3ms      | ×       | 51.3ms   | ×        |
| YOLOv4             | Detect           | GPU                | INT8                  | ×        | 195.7ms      | ×       | ?        | 8.7ms    |
| YOLOv5n            | Classify         | CPU                | FP32                  | 19.5ms   | 21.5ms       | 44.0ms  | 16.1ms   | ×        |
| YOLOv5n            | Classify         | GPU                | FP32                  | 13.7ms   | 14.6ms       | 11.3ms  | ?        | 1.0ms    |
| YOLOv5n            | Classify         | CPU                | FP16                  | ×        | 24.8ms       | 43.6ms  | 15.9ms   | ×        |
| YOLOv5n            | Classify         | GPU                | FP16                  | 14.5ms   | 16.4ms       | 11.2ms  | ?        | 0.9ms    |
| YOLOv5n            | Classify         | CPU                | INT8                  | ×        | 27.6ms       | ×       | 15.6ms   | ×        |
| YOLOv5n            | Classify         | GPU                | INT8                  | ×        | 28.3ms       | ×       | ?        | 0.9ms    |
| YOLOv5n            | Detect           | CPU                | FP32                  | 22.5ms   | 26.8ms       | 92.6ms  | 13.7ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP32                  | 13.2ms   | 9.7ms        | 9.5ms   | ?        | 2.7ms    |
| YOLOv5n            | Detect           | CPU                | FP16                  | ×        | 33.5ms       | 92.1ms  | 13.5ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP16                  | 12.8ms   | 14.6ms       | 8.9ms   | ?        | 2.8ms    |
| YOLOv5n            | Detect           | CPU                | INT8                  | ×        | 35.7ms       | ×       | 14.0ms   | ×        |
| YOLOv5n            | Detect           | GPU                | INT8                  | ×        | 31.1ms       | ×       | ?        | 3.6ms    |
| YOLOv5n            | Segment          | CPU                | FP32                  | 45.6ms   | 47.4ms       | 138.5ms | 39.0ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP32                  | 24.4ms   | 25.3ms       | 25.2ms  | ?        | 25.9ms   |
| YOLOv5n            | Segment          | CPU                | FP16                  | ×        | 56.2ms       | 138.0ms | 38.7ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP16                  | 28.3ms   | 33.3ms       | 18.3ms  | ?        | 25.9ms   |
| YOLOv5n            | Segment          | CPU                | INT8                  | ×        | 61.5ms       | ×       | 49.3ms   | ×        |
| YOLOv5n            | Segment          | GPU                | INT8                  | ×        | 54.8ms       | ×       | ?        | 19.9ms   |
| YOLOv6n            | Detect           | CPU                | FP32                  | 26.7ms   | 26.7ms       | 41.3ms  | 13.3ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP32                  | 11.5ms   | 10.0ms       | 7.4ms   | ?        | 3.9ms    |
| YOLOv6n            | Detect           | CPU                | FP16                  | ×        | 37.9ms       | 40.9ms  | 13.1ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP16                  | 10.9ms   | 18.0ms       | 8.4ms   | ?        | 4.0ms    |
| YOLOv6n            | Detect           | CPU                | INT8                  | ×        | 49.0ms       | ×       | 11.8ms   | ×        |
| YOLOv6n            | Detect           | GPU                | INT8                  | ×        | 43.4ms       | ×       | ?        | 3.4ms    |
| YOLOv7t            | Detect           | CPU                | FP32                  | 33.3ms   | 25.4ms       | 89.2ms  | 14.8ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP32                  | 13.3ms   | 9.7ms        | 8.4ms   | ?        | 2.9ms    |
| YOLOv7t            | Detect           | CPU                | FP16                  | ×        | 42.1ms       | 88.8ms  | 14.6ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP16                  | 13.2ms   | 16.8ms       | 8.0ms   | ?        | 2.6ms    |
| YOLOv7t            | Detect           | CPU                | INT8                  | ×        | 44.7ms       | ×       | 15.0ms   | ×        |
| YOLOv7t            | Detect           | GPU                | INT8                  | ×        | 39.8ms       | ×       | ?        | 2.6ms    |
| YOLOv8n            | Classify         | CPU                | FP32                  | 4.8ms    | 3.7ms        | 4.3ms   | 3.2ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP32                  | 1.5ms    | 1.2ms        | 2.3ms   | ?        | 0.9ms    |
| YOLOv8n            | Classify         | CPU                | FP16                  | ×        | 4.0ms        | 4.2ms   | 3.1ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP16                  | 1.5ms    | 1.5ms        | 2.5ms   | ?        | 0.8ms    |
| YOLOv8n            | Classify         | CPU                | INT8                  | ×        | 4.3ms        | ×       | 3.9ms    | ×        |
| YOLOv8n            | Classify         | GPU                | INT8                  | ×        | 3.9ms        | ×       | ?        | 0.8ms    |
| YOLOv8n            | Detect           | CPU                | FP32                  | 27.7ms   | 37.9ms       | 56.9ms  | 15.3ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP32                  | 11.5ms   | 10.4ms       | 9.9ms   | ?        | 4.0ms    |
| YOLOv8n            | Detect           | CPU                | FP16                  | ×        | 45.6ms       | 56.5ms  | 15.1ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP16                  | 11.8ms   | 18.3ms       | 9.9ms   | ?        | 3.6ms    |
| YOLOv8n            | Detect           | CPU                | INT8                  | ×        | 42.4ms       | ×       | 17.4ms   | ×        |
| YOLOv8n            | Detect           | GPU                | INT8                  | ×        | 39.6ms       | ×       | ?        | 3.5ms    |
| YOLOv8n            | Segment          | CPU                | FP32                  | 72.0ms   | 78.2ms       | 153.0ms | 37.9ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP32                  | 38.1ms   | 36.6ms       | 35.7ms  | ?        | 29.6ms   |
| YOLOv8n            | Segment          | CPU                | FP16                  | ×        | 91.4ms       | 152.6ms | 37.6ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP16                  | 35.2ms   | 45.1ms       | 35.2ms  | ?        | 28.4ms   |
| YOLOv8n            | Segment          | CPU                | INT8                  | ×        | 93.3ms       | ×       | 52.4ms   | ×        |
| YOLOv8n            | Segment          | GPU                | INT8                  | ×        | 77.8ms       | ×       | ?        | 25.0ms   |
| YOLOv9t            | Detect           | CPU                | FP32                  | 46.6ms   | 45.9ms       | 72.1ms  | 22.6ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP32                  | 14.6ms   | 12.4ms       | 14.1ms  | ?        | 5.0ms    |
| YOLOv9t            | Detect           | CPU                | FP16                  | ×        | 58.3ms       | 71.7ms  | 22.4ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP16                  | 14.9ms   | 18.1ms       | 16.7ms  | ?        | 4.4ms    |
| YOLOv9t            | Detect           | CPU                | INT8                  | ×        | 64.6ms       | ×       | 24.8ms   | ×        |
| YOLOv9t            | Detect           | GPU                | INT8                  | ×        | 55.7ms       | ×       | ?        | 4.5ms    |
| YOLOv9c            | Segment          | CPU                | FP32                  | 222.4ms  | 204.2ms      | 438.0ms | 114.9ms  | ×        |
| YOLOv9c            | Segment          | GPU                | FP32                  | 46.1ms   | 75.5ms       | 50.7ms  | ?        | 36.1ms   |
| YOLOv9c            | Segment          | CPU                | FP16                  | ×        | 241.8ms      | 437.6ms | 114.6ms  | ×        |
| YOLOv9c            | Segment          | GPU                | FP16                  | 46.0ms   | 76.8ms       | 43.1ms  | ?        | 33.8ms   |
| YOLOv9c            | Segment          | CPU                | INT8                  | ×        | 235.9ms      | ×       | 129.4ms  | ×        |
| YOLOv9c            | Segment          | GPU                | INT8                  | ×        | 233.6ms      | ×       | ?        | 25.2ms   |
| YOLOv10n           | Detect           | CPU                | FP32                  | 28.6ms   | 35.6ms       | 61.6ms  | 16.5ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP32                  | 11.3ms   | 11.5ms       | ×       | ?        | 4.1ms    |
| YOLOv10n           | Detect           | CPU                | FP16                  | ×        | 64.2ms       | 61.2ms  | 16.3ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP16                  | 12.0ms   | 17.2ms       | ×       | ?        | 3.9ms    |
| YOLOv10n           | Detect           | CPU                | INT8                  | ×        | 56.4ms       | ×       | 18.3ms   | ×        |
| YOLOv10n           | Detect           | GPU                | INT8                  | ×        | 49.2ms       | ×       | ?        | 3.7ms    |
| YOLOv11n           | Classify         | CPU                | FP32                  | 5.1ms    | 3.4ms        | 5.6ms   | 4.1ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP32                  | 1.8ms    | 1.5ms        | ×       | ?        | 1.0ms    |
| YOLOv11n           | Classify         | CPU                | FP16                  | ×        | 4.8ms        | 5.5ms   | 4.0ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP16                  | 2.1ms    | 1.8ms        | ×       | ?        | 0.9ms    |
| YOLOv11n           | Classify         | CPU                | INT8                  | ×        | 4.1ms        | ×       | 4.3ms    | ×        |
| YOLOv11n           | Classify         | GPU                | INT8                  | ×        | 4.9ms        | ×       | ?        | 0.9ms    |
| YOLOv11n           | Detect           | CPU                | FP32                  | 32.4ms   | 32.8ms       | 61.0ms  | 16.9ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP32                  | 11.6ms   | 10.6ms       | ×       | ?        | 4.1ms    |
| YOLOv11n           | Detect           | CPU                | FP16                  | ×        | 62.7ms       | 60.6ms  | 16.7ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP16                  | 11.9ms   | 18.8ms       | ×       | ?        | 3.7ms    |
| YOLOv11n           | Detect           | CPU                | INT8                  | ×        | 51.0ms       | ×       | 18.6ms   | ×        |
| YOLOv11n           | Detect           | GPU                | INT8                  | ×        | 45.0ms       | ×       | ?        | 3.6ms    |
| YOLOv11n           | Segment          | CPU                | FP32                  | 63.6ms   | 67.5ms       | 149.8ms | 38.2ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP32                  | 29.5ms   | 31.3ms       | ×       | ?        | 26.2ms   |
| YOLOv11n           | Segment          | CPU                | FP16                  | ×        | 82.3ms       | 149.4ms | 37.9ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP16                  | 30.4ms   | 41.7ms       | ×       | ?        | 25.3ms   |
| YOLOv11n           | Segment          | CPU                | INT8                  | ×        | 99.0ms       | ×       | 49.3ms   | ×        |
| YOLOv11n           | Segment          | GPU                | INT8                  | ×        | 73.9ms       | ×       | ?        | 22.5ms   |
| YOLOv12n           | Classify         | CPU                | FP32                  | 8.7ms    | 6.1ms        | 14.5ms  | 5.8ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP32                  | 2.8ms    | 2.2ms        | 5.2ms   | ?        | 1.2ms    |
| YOLOv12n           | Classify         | CPU                | FP16                  | ×        | 8.5ms        | 14.4ms  | 5.7ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP16                  | 2.9ms    | 2.5ms        | 6.0ms   | ?        | 1.2ms    |
| YOLOv12n           | Classify         | CPU                | INT8                  | ×        | 9.2ms        | ×       | 6.3ms    | ×        |
| YOLOv12n           | Classify         | GPU                | INT8                  | ×        | 8.4ms        | ×       | ?        | 1.3ms    |
| YOLOv12n           | Detect           | CPU                | FP32                  | 41.5ms   | 45.0ms       | 162.2ms | 22.9ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP32                  | 13.1ms   | 11.7ms       | 14.4ms  | ?        | 4.5ms    |
| YOLOv12n           | Detect           | CPU                | FP16                  | ×        | 79.8ms       | 161.8ms | 22.7ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP16                  | 13.6ms   | 19.9ms       | 15.5ms  | ?        | 4.1ms    |
| YOLOv12n           | Detect           | CPU                | INT8                  | ×        | 73.0ms       | ×       | 24.2ms   | ×        |
| YOLOv12n           | Detect           | GPU                | INT8                  | ×        | 60.9ms       | ×       | ?        | 4.4ms    |
| YOLOv12n           | Segment          | CPU                | FP32                  | 81.7ms   | 75.5ms       | 211.9ms | 45.3ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP32                  | 36.6ms   | 33.6ms       | 37.1ms  | ?        | 27.9ms   |
| YOLOv12n           | Segment          | CPU                | FP16                  | ×        | 98.2ms       | 211.5ms | 45.0ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP16                  | 33.4ms   | 42.1ms       | 36.0ms  | ?        | 24.6ms   |
| YOLOv12n           | Segment          | CPU                | INT8                  | ×        | 112.5ms      | ×       | 55.2ms   | ×        |
| YOLOv12n           | Segment          | GPU                | INT8                  | ×        | 92.1ms       | ×       | ?        | 21.4ms   |
| YOLOv13n           | Detect           | CPU                | FP32                  | 52.2ms   | 48.3ms       | 179.9ms | 25.3ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP32                  | 14.2ms   | 12.4ms       | 17.1ms  | ?        | 4.9ms    |
| YOLOv13n           | Detect           | CPU                | FP16                  | ×        | 104.5ms      | 179.5ms | 25.1ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP16                  | 14.6ms   | 20.4ms       | 19.9ms  | ?        | 4.4ms    |
| YOLOv13n           | Detect           | CPU                | INT8                  | ×        | 106.7ms      | ×       | 25.8ms   | ×        |
| YOLOv13n           | Detect           | GPU                | INT8                  | ×        | 87.1ms       | ×       | ?        | 5.3ms    |

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
