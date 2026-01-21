# yolo-inference
C++ and Python implementations of YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv13, YOLO26 inference.

Supported inference backends include Libtorch/PyTorch, ONNXRuntime, OpenCV, OpenVINO, TensorRT. 

Supported task types include Classify, Detect, Segment, Pose, OBB.

Supported model types include FP32, FP16, INT8.

Dependencies(tested):
* [CUDA](https://developer.nvidia.com/cuda-downloads) version 11.8.0/12.5.1/12.8.0
* [OpenCV](https://github.com/opencv/opencv) version 4.9.0/4.10.0/4.11.0 (built with CUDA)
* [ONNXRuntime](https://github.com/microsoft/onnxruntime) version 1.18.1/1.20.0/1.22.0
* [OpenVINO](https://github.com/openvinotoolkit/openvino) version 2024.1.0/2024.4.0/2025.2.0
* [TensorRT](https://developer.nvidia.com/tensorrt/download) version 8.2.1.8/10.6.0.26/10.8.0.43
* [Torch](https://pytorch.org) version 2.0.0+cu118/2.5.0+cu124/2.7.0+cu128

You can test C++ code with:
```bash
# Linux
mkdir build ; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cp ../test.sh .
bash ./test.sh
```

C++ test in Docker with 25 vCPU Intel(R) Xeon(R) Platinum 8470Q , RTX 5090(32GB):
|       Model        |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime  | OpenCV  | OpenVINO | TensorRT |
| :----------------: | :--------------: | :----------------: | :-------------------: | :------: | :----------: | :-----: | :------: | :------: |
| YOLOv3u            | Detect           | CPU                | FP32                  | 167.3ms  | 252.5ms      | 464.0ms | 86.6ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP32                  | 10.0ms   | 11.9ms       | 26.2ms  | ?        | 6.3ms    |
| YOLOv3u            | Detect           | CPU                | FP16                  | ×        | 298.6ms      | 460.2ms | 87.2ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP16                  | 7.7ms    | 18.3ms       | 7.6ms   | ?        | 2.4ms    |
| YOLOv3u            | Detect           | CPU                | INT8                  | ×        | 243.3ms      | ×       | 97.4ms   | ×        |
| YOLOv3u            | Detect           | GPU                | INT8                  | ×        | 237.1ms      | ×       | ?        | 1.7ms    |
| YOLOv4             | Detect           | CPU                | FP32                  | ?        | 285.7ms      | 284.6ms | 64.5ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP32                  | ?        | 11.5ms       | 19.0ms  | ?        | 4.8ms    |
| YOLOv4             | Detect           | CPU                | FP16                  | ×        | 342.3ms      | 281.5ms | 65.4ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP16                  | ?        | 20.4ms       | 11.3ms  | ?        | 2.3ms    |
| YOLOv4             | Detect           | CPU                | INT8                  | ×        | 319.1ms      | ×       | 65.6ms   | ×        |
| YOLOv4             | Detect           | GPU                | INT8                  | ×        | 189.2ms      | ×       | ?        | 1.9ms    |
| YOLOv5n            | Classify         | CPU                | FP32                  | 14.8ms   | 17.2ms       | 24.1ms  | 6.8ms    | ×        |
| YOLOv5n            | Classify         | GPU                | FP32                  | 5.0ms    | 8.5ms        | 4.6ms   | ?        | 0.5ms    |
| YOLOv5n            | Classify         | CPU                | FP16                  | ×        | 18.8ms       | 23.8ms  | 6.8ms    | ×        |
| YOLOv5n            | Classify         | GPU                | FP16                  | 7.0ms    | 11.0ms       | 4.5ms   | ?        | 0.4ms    |
| YOLOv5n            | Classify         | CPU                | INT8                  | ×        | 22.5ms       | ×       | 7.1ms    | ×        |
| YOLOv5n            | Classify         | GPU                | INT8                  | ×        | 21.6ms       | ×       | ?        | 0.4ms    |
| YOLOv5n            | Detect           | CPU                | FP32                  | 22.6ms   | 23.3ms       | 87.7ms  | 11.8ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP32                  | 4.9ms    | 7.7ms        | 7.1ms   | ?        | 1.1ms    |
| YOLOv5n            | Detect           | CPU                | FP16                  | ×        | 35.1ms       | 87.2ms  | 11.7ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP16                  | 4.6ms    | 14.3ms       | 6.5ms   | ?        | 0.9ms    |
| YOLOv5n            | Detect           | CPU                | INT8                  | ×        | 31.4ms       | ×       | 12.5ms   | ×        |
| YOLOv5n            | Detect           | GPU                | INT8                  | ×        | 27.4ms       | ×       | ?        | 0.8ms    |
| YOLOv5n            | Segment          | CPU                | FP32                  | 27.2ms   | 31.4ms       | 116.6ms | 14.1ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP32                  | 7.9ms    | 9.7ms        | 9.2ms   | ?        | 1.3ms    |
| YOLOv5n            | Segment          | CPU                | FP16                  | ×        | 42.8ms       | 116.1ms | 14.6ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP16                  | 7.6ms    | 19.2ms       | 8.7ms   | ?        | 1.0ms    |
| YOLOv5n            | Segment          | CPU                | INT8                  | ×        | 39.7ms       | ×       | 15.9ms   | ×        |
| YOLOv5n            | Segment          | GPU                | INT8                  | ×        | 33.5ms       | ×       | ?        | 1.0ms    |
| YOLOv6n            | Detect           | CPU                | FP32                  | 24.0ms   | 19.8ms       | 36.3ms  | 11.4ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP32                  | 4.6ms    | 8.1ms        | 5.4ms   | ?        | 1.1ms    |
| YOLOv6n            | Detect           | CPU                | FP16                  | ×        | 38.5ms       | 35.9ms  | 11.9ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP16                  | 4.5ms    | 17.0ms       | 6.3ms   | ?        | 0.8ms    |
| YOLOv6n            | Detect           | CPU                | INT8                  | ×        | 47.0ms       | ×       | 11.8ms   | ×        |
| YOLOv6n            | Detect           | GPU                | INT8                  | ×        | 42.9ms       | ×       | ?        | 0.8ms    |
| YOLOv7t            | Detect           | CPU                | FP32                  | 29.8ms   | 19.7ms       | 86.9ms  | 13.1ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP32                  | 5.6ms    | 7.8ms        | 6.0ms   | ?        | 1.3ms    |
| YOLOv7t            | Detect           | CPU                | FP16                  | ×        | 45.2ms       | 86.5ms  | 12.6ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP16                  | 5.1ms    | 16.9ms       | 5.5ms   | ?        | 1.0ms    |
| YOLOv7t            | Detect           | CPU                | INT8                  | ×        | 38.5ms       | ×       | 14.2ms   | ×        |
| YOLOv7t            | Detect           | GPU                | INT8                  | ×        | 39.6ms       | ×       | ?        | 0.5ms    |
| YOLOv8n            | Classify         | CPU                | FP32                  | 5.2ms    | 3.1ms        | 5.0ms   | 3.1ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP32                  | 1.2ms    | 1.1ms        | 1.8ms   | ?        | 0.4ms    |
| YOLOv8n            | Classify         | CPU                | FP16                  | ×        | 4.0ms        | 4.9ms   | 3.5ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP16                  | 1.3ms    | 1.4ms        | 1.9ms   | ?        | 0.4ms    |
| YOLOv8n            | Classify         | CPU                | INT8                  | ×        | 3.7ms        | ×       | 3.0ms    | ×        |
| YOLOv8n            | Classify         | GPU                | INT8                  | ×        | 3.6ms        | ×       | ?        | 0.6ms    |
| YOLOv8n            | Detect           | CPU                | FP32                  | 24.6ms   | 34.4ms       | 64.6ms  | 13.6ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP32                  | 4.7ms    | 8.2ms        | 6.2ms   | ?        | 1.3ms    |
| YOLOv8n            | Detect           | CPU                | FP16                  | ×        | 43.8ms       | 64.2ms  | 13.2ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP16                  | 4.5ms    | 17.1ms       | 6.0ms   | ?        | 0.9ms    |
| YOLOv8n            | Detect           | CPU                | INT8                  | ×        | 43.2ms       | ×       | 16.2ms   | ×        |
| YOLOv8n            | Detect           | GPU                | INT8                  | ×        | 38.1ms       | ×       | ?        | 0.8ms    |
| YOLOv8n            | Segment          | CPU                | FP32                  | 33.0ms   | 39.2ms       | 70.0ms  | 16.2ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP32                  | 7.3ms    | 10.1ms       | 8.3ms   | ?        | 1.7ms    |
| YOLOv8n            | Segment          | CPU                | FP16                  | ×        | 51.5ms       | 69.6ms  | 16.5ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP16                  | 7.5ms    | 21.5ms       | 8.4ms   | ?        | 1.3ms    |
| YOLOv8n            | Segment          | CPU                | INT8                  | ×        | 49.2ms       | ×       | 20.4ms   | ×        |
| YOLOv8n            | Segment          | GPU                | INT8                  | ×        | 42.4ms       | ×       | ?        | 1.1ms    |
| YOLOv8n            | Pose             | CPU                | FP32                  | 25.7ms   | 28.7ms       | 48.4ms  | 13.4ms   | ×        |
| YOLOv8n            | Pose             | GPU                | FP32                  | 4.1ms    | 8.7ms        | 6.1ms   | ?        | 1.2ms    |
| YOLOv8n            | Pose             | CPU                | FP16                  | ×        | 45.0ms       | 48.8ms  | 13.2ms   | ×        |
| YOLOv8n            | Pose             | GPU                | FP16                  | 4.0ms    | 12.4ms       | 6.3ms   | ?        | 0.9ms    |
| YOLOv8n            | Pose             | CPU                | INT8                  | ×        | 40.8ms       | ×       | 15.6ms   | ×        |
| YOLOv8n            | Pose             | GPU                | INT8                  | ×        | 35.6ms       | ×       | ?        | 0.8ms    |
| YOLOv8n            | OBB              | CPU                | FP32                  | 221.5ms  | 248.2ms      | 319.3ms | 174.5ms  | ×        |
| YOLOv8n            | OBB              | GPU                | FP32                  | 183.4ms  | 189.0ms      | 179.0ms | ?        | 2.9ms    |
| YOLOv8n            | OBB              | CPU                | FP16                  | ×        | ×            | 316.6ms | 177.1ms  | ×        |
| YOLOv8n            | OBB              | GPU                | FP16                  | 191.3ms  | ×            | 180.3ms | ?        | 2.3ms    |
| YOLOv8n            | OBB              | CPU                | INT8                  | ×        | 264.6ms      | ×       | 189.3ms  | ×        |
| YOLOv8n            | OBB              | GPU                | INT8                  | ×        | 244.3ms      | ×       | ?        | 2.2ms    |
| YOLOv9t            | Detect           | CPU                | FP32                  | 43.8ms   | 49.0ms       | 61.0ms  | 18.0ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP32                  | 7.5ms    | 10.3ms       | 11.2ms  | ?        | 2.2ms    |
| YOLOv9t            | Detect           | CPU                | FP16                  | ×        | 52.3ms       | 60.6ms  | 17.9ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP16                  | 7.9ms    | 16.6ms       | 12.8ms  | ?        | 1.7ms    |
| YOLOv9t            | Detect           | CPU                | INT8                  | ×        | 56.9ms       | ×       | 25.5ms   | ×        |
| YOLOv9t            | Detect           | GPU                | INT8                  | ×        | 45.4ms       | ×       | ?        | 1.9ms    |
| YOLOv9c            | Segment          | CPU                | FP32                  | 137.4ms  | 175.1ms      | 308.9ms | 60.5ms   | ×        |
| YOLOv9c            | Segment          | GPU                | FP32                  | 10.9ms   | 27.5ms       | 19.0ms  | ?        | 4.8ms    |
| YOLOv9c            | Segment          | CPU                | FP16                  | ×        | 182.6ms      | 308.5ms | 60.8ms   | ×        |
| YOLOv9c            | Segment          | GPU                | FP16                  | 9.5ms    | 48.0ms       | 12.6ms  | ?        | 2.4ms    |
| YOLOv9c            | Segment          | CPU                | INT8                  | ×        | 178.1ms      | ×       | 68.3ms   | ×        |
| YOLOv9c            | Segment          | GPU                | INT8                  | ×        | 159.0ms      | ×       | ?        | 1.9ms    |
| YOLOv10n           | Detect           | CPU                | FP32                  | 25.9ms   | 30.0ms       | 54.7ms  | 14.4ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP32                  | 4.9ms    | 9.0ms        | ×       | ?        | 1.3ms    |
| YOLOv10n           | Detect           | CPU                | FP16                  | ×        | 55.3ms       | 54.3ms  | 14.1ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP16                  | 4.8ms    | 16.2ms       | ×       | ?        | 1.0ms    |
| YOLOv10n           | Detect           | CPU                | INT8                  | ×        | 50.0ms       | ×       | 16.7ms   | ×        |
| YOLOv10n           | Detect           | GPU                | INT8                  | ×        | 44.2ms       | ×       | ?        | 1.0ms    |
| YOLOv11n           | Classify         | CPU                | FP32                  | 5.7ms    | 3.1ms        | 5.0ms   | 4.0ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP32                  | 1.6ms    | 1.3ms        | ×       | ?        | 0.6ms    |
| YOLOv11n           | Classify         | CPU                | FP16                  | ×        | 4.3ms        | 4.9ms   | 4.2ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP16                  | 1.6ms    | 1.7ms        | ×       | ?        | 0.5ms    |
| YOLOv11n           | Classify         | CPU                | INT8                  | ×        | 4.2ms        | ×       | 5.2ms    | ×        |
| YOLOv11n           | Classify         | GPU                | INT8                  | ×        | 4.4ms        | ×       | ?        | 0.5ms    |
| YOLOv11n           | Detect           | CPU                | FP32                  | 29.6ms   | 30.2ms       | 57.9ms  | 14.8ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP32                  | 5.1ms    | 8.4ms        | ×       | ?        | 1.4ms    |
| YOLOv11n           | Detect           | CPU                | FP16                  | ×        | 55.5ms       | 57.5ms  | 14.7ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP16                  | 5.2ms    | 17.6ms       | ×       | ?        | 1.1ms    |
| YOLOv11n           | Detect           | CPU                | INT8                  | ×        | 42.6ms       | ×       | 17.1ms   | ×        |
| YOLOv11n           | Detect           | GPU                | INT8                  | ×        | 38.1ms       | ×       | ?        | 1.0ms    |
| YOLOv11n           | Segment          | CPU                | FP32                  | 35.8ms   | 39.8ms       | 77.9ms  | 18.5ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP32                  | 8.0ms    | 11.8ms       | ×       | ?        | 1.8ms    |
| YOLOv11n           | Segment          | CPU                | FP16                  | ×        | 68.0ms       | 77.5ms  | 18.2ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP16                  | 8.7ms    | 21.5ms       | ×       | ?        | 1.4ms    |
| YOLOv11n           | Segment          | CPU                | INT8                  | ×        | 55.5ms       | ×       | 21.3ms   | ×        |
| YOLOv11n           | Segment          | GPU                | INT8                  | ×        | 47.9ms       | ×       | ?        | 1.3ms    |
| YOLOv11n           | Pose             | CPU                | FP32                  | 32.6ms   | 35.5ms       | 52.1ms  | 15.2ms   | ×        |
| YOLOv11n           | Pose             | GPU                | FP32                  | 5.1ms    | 9.2ms        | ×       | ?        | 1.4ms    |
| YOLOv11n           | Pose             | CPU                | FP16                  | ×        | 53.1ms       | 51.7ms  | 13.9ms   | ×        |
| YOLOv11n           | Pose             | GPU                | FP16                  | 4.6ms    | 12.5ms       | ×       | ?        | 1.0ms    |
| YOLOv11n           | Pose             | CPU                | INT8                  | ×        | 53.9ms       | ×       | 16.2ms   | ×        |
| YOLOv11n           | Pose             | GPU                | INT8                  | ×        | 47.6ms       | ×       | ?        | 0.9ms    |
| YOLOv11n           | OBB              | CPU                | FP32                  | 200.6ms  | 232.1ms      | 318.0ms | 156.3ms  | ×        |
| YOLOv11n           | OBB              | GPU                | FP32                  | 155.8ms  | 161.4ms      | ×       | ?        | 2.9ms    |
| YOLOv11n           | OBB              | CPU                | FP16                  | ×        | ×            | 320.0ms | 156.6ms  | ×        |
| YOLOv11n           | OBB              | GPU                | FP16                  | 161.5ms  | ×            | ×       | ?        | 2.3ms    |
| YOLOv11n           | OBB              | CPU                | INT8                  | ×        | 239.8ms      | ×       | 155.1ms  | ×        |
| YOLOv11n           | OBB              | GPU                | INT8                  | ×        | 230.1ms      | ×       | ?        | 2.2ms    |
| YOLOv12n           | Classify         | CPU                | FP32                  | 9.4ms    | 4.7ms        | 12.5ms  | 9.4ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP32                  | 2.5ms    | 1.9ms        | 4.5ms   | ?        | 0.8ms    |
| YOLOv12n           | Classify         | CPU                | FP16                  | ×        | 7.9ms        | 12.4ms  | 9.6ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP16                  | 2.6ms    | 2.2ms        | 5.3ms   | ?        | 0.6ms    |
| YOLOv12n           | Classify         | CPU                | INT8                  | ×        | 7.6ms        | ×       | 11.4ms   | ×        |
| YOLOv12n           | Classify         | GPU                | INT8                  | ×        | 7.6ms        | ×       | ?        | 0.8ms    |
| YOLOv12n           | Detect           | CPU                | FP32                  | 39.2ms   | 39.0ms       | 158.0ms | 17.8ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP32                  | 6.6ms    | 9.4ms        | 10.5ms  | ?        | 1.7ms    |
| YOLOv12n           | Detect           | CPU                | FP16                  | ×        | 71.2ms       | 157.6ms | 18.4ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP16                  | 6.7ms    | 19.1ms       | 11.3ms  | ?        | 1.3ms    |
| YOLOv12n           | Detect           | CPU                | INT8                  | ×        | 61.3ms       | ×       | 21.2ms   | ×        |
| YOLOv12n           | Detect           | GPU                | INT8                  | ×        | 52.1ms       | ×       | ?        | 1.7ms    |
| YOLOv12n           | Segment          | CPU                | FP32                  | 47.6ms   | 47.8ms       | 191.6ms | 23.2ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP32                  | 9.6ms    | 12.5ms       | 13.2ms  | ?        | 2.1ms    |
| YOLOv12n           | Segment          | CPU                | FP16                  | ×        | 84.2ms       | 191.2ms | 23.4ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP16                  | 9.6ms    | 22.0ms       | 13.6ms  | ?        | 1.6ms    |
| YOLOv12n           | Segment          | CPU                | INT8                  | ×        | 74.4ms       | ×       | 27.3ms   | ×        |
| YOLOv12n           | Segment          | GPU                | INT8                  | ×        | 57.6ms       | ×       | ?        | 2.0ms    |
| YOLOv12n           | Pose             | CPU                | FP32                  | 40.4ms   | 40.0ms       | 136.6ms | 19.7ms   | ×        |
| YOLOv12n           | Pose             | GPU                | FP32                  | 6.0ms    | 9.8ms        | 10.1ms  | ?        | 1.7ms    |
| YOLOv12n           | Pose             | CPU                | FP16                  | ×        | ×            | 138.5ms | 17.5ms   | ×        |
| YOLOv12n           | Pose             | GPU                | FP16                  | 5.8ms    | ×            | 10.2ms  | ?        | 1.3ms    |
| YOLOv12n           | Pose             | CPU                | INT8                  | ×        | 71.2ms       | ×       | 22.0ms   | ×        |
| YOLOv12n           | Pose             | GPU                | INT8                  | ×        | 57.7ms       | ×       | ?        | 1.4ms    |
| YOLOv12n           | OBB              | CPU                | FP32                  | 197.7ms  | 170.2ms      | 515.4ms | 78.9ms   | ×        |
| YOLOv12n           | OBB              | GPU                | FP32                  | 82.9ms   | 92.6ms       | 86.0ms  | ?        | 3.8ms    |
| YOLOv12n           | OBB              | CPU                | FP16                  | ×        | ×            | 510.2ms | 79.4ms   | ×        |
| YOLOv12n           | OBB              | GPU                | FP16                  | 82.2ms   | ×            | 86.5ms  | ?        | 3.1ms    |
| YOLOv12n           | OBB              | CPU                | INT8                  | ×        | 244.5ms      | ×       | 80.8ms   | ×        |
| YOLOv12n           | OBB              | GPU                | INT8                  | ×        | 181.0ms      | ×       | ?        | 3.3ms    |
| YOLOv13n           | Detect           | CPU                | FP32                  | 50.1ms   | 45.7ms       | 183.1ms | 21.7ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP32                  | 7.9ms    | 9.9ms        | 14.3ms  | ?        | 2.1ms    |
| YOLOv13n           | Detect           | CPU                | FP16                  | ×        | 97.6ms       | 182.7ms | 21.5ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP16                  | 7.9ms    | 19.2ms       | 16.4ms  | ?        | 1.7ms    |
| YOLOv13n           | Detect           | CPU                | INT8                  | ×        | 109.2ms      | ×       | 22.2ms   | ×        |
| YOLOv13n           | Detect           | GPU                | INT8                  | ×        | 84.4ms       | ×       | ?        | 2.6ms    |
| YOLO26n            | Classify         | CPU                | FP32                  | 5.8ms    | 3.6ms        | 5.7ms   | 4.9ms    | ×        |
| YOLO26n            | Classify         | GPU                | FP32                  | 1.6ms    | 4.0ms        | ×       | ?        | 0.6ms    |
| YOLO26n            | Classify         | CPU                | FP16                  | ×        | 5.2ms        | 4.7ms   | 4.0ms    | ×        |
| YOLO26n            | Classify         | GPU                | FP16                  | 1.6ms    | 5.7ms        | ×       | ?        | 0.5ms    |
| YOLO26n            | Classify         | CPU                | INT8                  | ×        | 4.7ms        | ×       | 4.7ms    | ×        |
| YOLO26n            | Classify         | GPU                | INT8                  | ×        | 4.6ms        | ×       | ?        | 0.5ms    |
| YOLO26n            | Detect           | CPU                | FP32                  | 30.8ms   | 32.0ms       | 64.8ms  | 14.1ms   | ×        |
| YOLO26n            | Detect           | GPU                | FP32                  | 5.0ms    | 31.7ms       | ×       | ?        | 1.5ms    |
| YOLO26n            | Detect           | CPU                | FP16                  | ×        | ×            | 63.5ms  | 14.3ms   | ×        |
| YOLO26n            | Detect           | GPU                | FP16                  | 4.9ms    | ×            | ×       | ?        | 1.1ms    |
| YOLO26n            | Detect           | CPU                | INT8                  | ×        | 50.4ms       | ×       | 15.7ms   | ×        |
| YOLO26n            | Detect           | GPU                | INT8                  | ×        | 47.8ms       | ×       | ?        | 1.2ms    |
| YOLO26n            | Segment          | CPU                | FP32                  | 37.7ms   | 46.4ms       | 79.3ms  | 19.6ms   | ×        |
| YOLO26n            | Segment          | GPU                | FP32                  | 7.8ms    | 49.7ms       | ×       | ?        | 1.9ms    |
| YOLO26n            | Segment          | CPU                | FP16                  | ×        | ×            | 83.5ms  | 19.3ms   | ×        |
| YOLO26n            | Segment          | GPU                | FP16                  | 7.6ms    | ×            | ×       | ?        | 1.5ms    |
| YOLO26n            | Segment          | CPU                | INT8                  | ×        | 70.8ms       | ×       | 22.8ms   | ×        |
| YOLO26n            | Segment          | GPU                | INT8                  | ×        | 64.5ms       | ×       | ?        | 1.5ms    |
| YOLO26n            | Pose             | CPU                | FP32                  | 33.4ms   | 33.6ms       | ×       | 15.0ms   | ×        |
| YOLO26n            | Pose             | GPU                | FP32                  | 5.4ms    | 37.7ms       | ×       | ?        | 1.4ms    |
| YOLO26n            | Pose             | CPU                | FP16                  | ×        | ×            | ×       | 15.3ms   | ×        |
| YOLO26n            | Pose             | GPU                | FP16                  | 5.3ms    | ×            | ×       | ?        | 1.1ms    |
| YOLO26n            | Pose             | CPU                | INT8                  | ×        | 52.9ms       | ×       | 16.8ms   | ×        |
| YOLO26n            | Pose             | GPU                | INT8                  | ×        | 49.0ms       | ×       | ?        | 1.0ms    |
| YOLO26n            | OBB              | CPU                | FP32                  | 48.8ms   | 71.6ms       | 184.8ms | 30.1ms   | ×        |
| YOLO26n            | OBB              | GPU                | FP32                  | 8.2ms    | 78.7ms       | ×       | ?        | 3.1ms    |
| YOLO26n            | OBB              | CPU                | FP16                  | ×        | ×            | 174.3ms | 31.2ms   | ×        |
| YOLO26n            | OBB              | GPU                | FP16                  | 8.6ms    | ×            | ×       | ?        | 2.6ms    |
| YOLO26n            | OBB              | CPU                | INT8                  | ×        | 104.3ms      | ×       | 33.5ms   | ×        |
| YOLO26n            | OBB              | GPU                | INT8                  | ×        | 97.1ms       | ×       | ?        | 2.5ms    |

You can test Python code with:
```bash
# Linux
pip install -r requirements.txt
./run.sh
```

Python test in Docker with 25 vCPU Intel(R) Xeon(R) Platinum 8470Q , RTX 5090(32GB):
|       Model        |       Task       |       Device       |       Precision       | PyTorch  | ONNXRuntime  | OpenCV  | OpenVINO | TensorRT |
| :----------------: | :--------------: | :----------------: | :-------------------: | :------: | :----------: | :-----: | :------: | :------: |
| YOLOv3u            | Detect           | CPU                | FP32                  | 172.8ms  | 262.4ms      | 524.5ms | 71.4ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP32                  | 11.2ms   | 17.0ms       | 28.1ms  | ?        | 9.1ms    |
| YOLOv3u            | Detect           | CPU                | FP16                  | ×        | 369.1ms      | 540.4ms | 69.4ms   | ×        |
| YOLOv3u            | Detect           | GPU                | FP16                  | 9.4ms    | 22.4ms       | 9.9ms   | ?        | 4.9ms    |
| YOLOv3u            | Detect           | CPU                | INT8                  | ×        | 297.5ms      | ×       | 58.6ms   | ×        |
| YOLOv3u            | Detect           | GPU                | INT8                  | ×        | 278.6ms      | ×       | ?        | 4.4ms    |
| YOLOv4             | Detect           | CPU                | FP32                  | ×        | 290.8ms      | 285.5ms | 51.5ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP32                  | ×        | 17.9ms       | 22.0ms  | ?        | 10.2ms   |
| YOLOv4             | Detect           | CPU                | FP16                  | ×        | 366.2ms      | 279.3ms | 51.2ms   | ×        |
| YOLOv4             | Detect           | GPU                | FP16                  | ×        | 24.7ms       | 15.1ms  | ?        | 8.5ms    |
| YOLOv4             | Detect           | CPU                | INT8                  | ×        | 323.1ms      | ×       | 52.8ms   | ×        |
| YOLOv4             | Detect           | GPU                | INT8                  | ×        | 191.4ms      | ×       | ?        | 7.9ms    |
| YOLOv5n            | Classify         | CPU                | FP32                  | 18.9ms   | 22.9ms       | 46.7ms  | 15.9ms   | ×        |
| YOLOv5n            | Classify         | GPU                | FP32                  | 14.7ms   | 14.4ms       | 11.6ms  | ?        | 1.0ms    |
| YOLOv5n            | Classify         | CPU                | FP16                  | ×        | 27.2ms       | 47.3ms  | 16.2ms   | ×        |
| YOLOv5n            | Classify         | GPU                | FP16                  | 14.1ms   | 16.2ms       | 11.1ms  | ?        | 0.9ms    |
| YOLOv5n            | Classify         | CPU                | INT8                  | ×        | 29.8ms       | ×       | 15.7ms   | ×        |
| YOLOv5n            | Classify         | GPU                | INT8                  | ×        | 28.1ms       | ×       | ?        | 0.9ms    |
| YOLOv5n            | Detect           | CPU                | FP32                  | 24.7ms   | 30.7ms       | 93.2ms  | 14.3ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP32                  | 8.7ms    | 9.8ms        | 9.4ms   | ?        | 2.8ms    |
| YOLOv5n            | Detect           | CPU                | FP16                  | ×        | 38.6ms       | 93.4ms  | 14.0ms   | ×        |
| YOLOv5n            | Detect           | GPU                | FP16                  | 8.8ms    | 14.9ms       | 8.9ms   | ?        | 3.3ms    |
| YOLOv5n            | Detect           | CPU                | INT8                  | ×        | 37.6ms       | ×       | 14.6ms   | ×        |
| YOLOv5n            | Detect           | GPU                | INT8                  | ×        | 33.5ms       | ×       | ?        | ×        |
| YOLOv5n            | Segment          | CPU                | FP32                  | 38.9ms   | 54.9ms       | 141.0ms | 39.1ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP32                  | 24.4ms   | 29.0ms       | 19.7ms  | ?        | 2.7ms    |
| YOLOv5n            | Segment          | CPU                | FP16                  | ×        | 65.0ms       | 133.1ms | 33.9ms   | ×        |
| YOLOv5n            | Segment          | GPU                | FP16                  | 24.9ms   | 38.5ms       | 18.1ms  | ?        | 3.4ms    |
| YOLOv5n            | Segment          | CPU                | INT8                  | ×        | 63.7ms       | ×       | 50.2ms   | ×        |
| YOLOv5n            | Segment          | GPU                | INT8                  | ×        | 56.2ms       | ×       | ?        | 3.4ms    |
| YOLOv6n            | Detect           | CPU                | FP32                  | 26.1ms   | 25.5ms       | 45.5ms  | 12.9ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP32                  | 9.0ms    | 9.9ms        | 8.5ms   | ?        | 3.6ms    |
| YOLOv6n            | Detect           | CPU                | FP16                  | ×        | 43.9ms       | 44.0ms  | 14.0ms   | ×        |
| YOLOv6n            | Detect           | GPU                | FP16                  | 9.2ms    | 18.0ms       | 10.4ms  | ?        | 3.2ms    |
| YOLOv6n            | Detect           | CPU                | INT8                  | ×        | 50.6ms       | ×       | 11.1ms   | ×        |
| YOLOv6n            | Detect           | GPU                | INT8                  | ×        | 50.3ms       | ×       | ?        | 3.3ms    |
| YOLOv7t            | Detect           | CPU                | FP32                  | ×        | 26.0ms       | 90.9ms  | 15.9ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP32                  | ×        | 9.8ms        | 8.5ms   | ?        | 3.1ms    |
| YOLOv7t            | Detect           | CPU                | FP16                  | ×        | 50.0ms       | 94.2ms  | 15.7ms   | ×        |
| YOLOv7t            | Detect           | GPU                | FP16                  | ×        | 16.7ms       | 8.0ms   | ?        | 2.5ms    |
| YOLOv7t            | Detect           | CPU                | INT8                  | ×        | 44.6ms       | ×       | 15.1ms   | ×        |
| YOLOv7t            | Detect           | GPU                | INT8                  | ×        | 42.2ms       | ×       | ?        | 2.6ms    |
| YOLOv8n            | Classify         | CPU                | FP32                  | 5.0ms    | 3.5ms        | 5.1ms   | 3.6ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP32                  | 1.4ms    | 1.2ms        | 2.4ms   | ?        | 0.9ms    |
| YOLOv8n            | Classify         | CPU                | FP16                  | ×        | 5.0ms        | 4.6ms   | 2.7ms    | ×        |
| YOLOv8n            | Classify         | GPU                | FP16                  | 1.5ms    | 1.4ms        | 2.5ms   | ?        | 0.8ms    |
| YOLOv8n            | Classify         | CPU                | INT8                  | ×        | 4.0ms        | ×       | 3.2ms    | ×        |
| YOLOv8n            | Classify         | GPU                | INT8                  | ×        | 3.8ms        | ×       | ?        | 0.8ms    |
| YOLOv8n            | Detect           | CPU                | FP32                  | 26.1ms   | 39.9ms       | 68.3ms  | 17.5ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP32                  | 9.3ms    | 10.1ms       | 9.6ms   | ?        | 3.9ms    |
| YOLOv8n            | Detect           | CPU                | FP16                  | ×        | 49.9ms       | 60.8ms  | 17.5ms   | ×        |
| YOLOv8n            | Detect           | GPU                | FP16                  | 9.3ms    | 17.9ms       | 10.0ms  | ?        | 3.6ms    |
| YOLOv8n            | Detect           | CPU                | INT8                  | ×        | 46.4ms       | ×       | 19.3ms   | ×        |
| YOLOv8n            | Detect           | GPU                | INT8                  | ×        | 39.7ms       | ×       | ?        | 3.4ms    |
| YOLOv8n            | Segment          | CPU                | FP32                  | 49.3ms   | 80.7ms       | 144.4ms | 51.7ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP32                  | 33.7ms   | 38.3ms       | 36.6ms  | ?        | 33.2ms   |
| YOLOv8n            | Segment          | CPU                | FP16                  | ×        | 95.9ms       | 150.3ms | 46.3ms   | ×        |
| YOLOv8n            | Segment          | GPU                | FP16                  | 33.0ms   | 50.6ms       | 37.9ms  | ?        | 32.4ms   |
| YOLOv8n            | Segment          | CPU                | INT8                  | ×        | 80.8ms       | ×       | 62.4ms   | ×        |
| YOLOv8n            | Segment          | GPU                | INT8                  | ×        | 65.9ms       | ×       | ?        | 23.3ms   |
| YOLOv8n            | Pose             | CPU                | FP32                  | 28.2ms   | 223.4ms      | 49.8ms  | 15.7ms   | ×        |
| YOLOv8n            | Pose             | GPU                | FP32                  | 10.0ms   | 9.7ms        | 8.9ms   | ?        | 2.6ms    |
| YOLOv8n            | Pose             | CPU                | FP16                  | ×        | 235.1ms      | 48.1ms  | 15.8ms   | ×        |
| YOLOv8n            | Pose             | GPU                | FP16                  | 10.1ms   | 12.4ms       | 7.2ms   | ?        | 2.2ms    |
| YOLOv8n            | Pose             | CPU                | INT8                  | ×        | 287.5ms      | ×       | 18.4ms   | ×        |
| YOLOv8n            | Pose             | GPU                | INT8                  | ×        | 215.3ms      | ×       | ?        | 2.2ms    |
| YOLOv8n            | OBB              | CPU                | FP32                  | 60.6ms   | 227.9ms      | 340.0ms | 219.4ms  | ×        |
| YOLOv8n            | OBB              | GPU                | FP32                  | 20.2ms   | 159.9ms      | 190.5ms | ?        | 141.4ms  |
| YOLOv8n            | OBB              | CPU                | FP16                  | ×        | ×            | 322.0ms | 209.9ms  | ×        |
| YOLOv8n            | OBB              | GPU                | FP16                  | 19.4ms   | ×            | 186.2ms | ?        | 178.7ms  |
| YOLOv8n            | OBB              | CPU                | INT8                  | ×        | 229.3ms      | ×       | 208.7ms  | ×        |
| YOLOv8n            | OBB              | GPU                | INT8                  | ×        | 216.8ms      | ×       | ?        | 180.3ms  |
| YOLOv9t            | Detect           | CPU                | FP32                  | 50.9ms   | 61.0ms       | 80.4ms  | 24.8ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP32                  | 12.8ms   | 12.5ms       | 14.3ms  | ?        | 4.6ms    |
| YOLOv9t            | Detect           | CPU                | FP16                  | ×        | 59.1ms       | 85.0ms  | 24.8ms   | ×        |
| YOLOv9t            | Detect           | GPU                | FP16                  | 12.8ms   | 17.6ms       | 16.7ms  | ?        | 4.3ms    |
| YOLOv9t            | Detect           | CPU                | INT8                  | ×        | 82.5ms       | ×       | 26.1ms   | ×        |
| YOLOv9t            | Detect           | GPU                | INT8                  | ×        | 69.6ms       | ×       | ?        | 4.5ms    |
| YOLOv9c            | Segment          | CPU                | FP32                  | 169.4ms  | 216.7ms      | 396.3ms | 117.8ms  | ×        |
| YOLOv9c            | Segment          | GPU                | FP32                  | 36.1ms   | 67.8ms       | 46.4ms  | ?        | 35.8ms   |
| YOLOv9c            | Segment          | CPU                | FP16                  | ×        | 288.3ms      | 400.0ms | 147.5ms  | ×        |
| YOLOv9c            | Segment          | GPU                | FP16                  | 34.9ms   | 84.1ms       | 37.1ms  | ?        | 32.7ms   |
| YOLOv9c            | Segment          | CPU                | INT8                  | ×        | 267.7ms      | ×       | 128.3ms  | ×        |
| YOLOv9c            | Segment          | GPU                | INT8                  | ×        | 228.1ms      | ×       | ?        | 28.9ms   |
| YOLOv10n           | Detect           | CPU                | FP32                  | 28.5ms   | 40.5ms       | 61.9ms  | 17.9ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP32                  | 9.5ms    | 11.3ms       | ×       | ?        | 3.9ms    |
| YOLOv10n           | Detect           | CPU                | FP16                  | ×        | 60.3ms       | 71.6ms  | 17.8ms   | ×        |
| YOLOv10n           | Detect           | GPU                | FP16                  | 9.3ms    | 17.1ms       | ×       | ?        | 3.6ms    |
| YOLOv10n           | Detect           | CPU                | INT8                  | ×        | 63.4ms       | ×       | 19.4ms   | ×        |
| YOLOv10n           | Detect           | GPU                | INT8                  | ×        | 52.8ms       | ×       | ?        | 3.5ms    |
| YOLOv11n           | Classify         | CPU                | FP32                  | 5.8ms    | 3.7ms        | 6.0ms   | 3.9ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP32                  | 1.8ms    | 1.6ms        | ×       | ?        | 1.0ms    |
| YOLOv11n           | Classify         | CPU                | FP16                  | ×        | 4.6ms        | 6.1ms   | 4.0ms    | ×        |
| YOLOv11n           | Classify         | GPU                | FP16                  | 2.0ms    | 1.8ms        | ×       | ?        | 0.9ms    |
| YOLOv11n           | Classify         | CPU                | INT8                  | ×        | 4.7ms        | ×       | 4.4ms    | ×        |
| YOLOv11n           | Classify         | GPU                | INT8                  | ×        | 5.1ms        | ×       | ?        | 0.9ms    |
| YOLOv11n           | Detect           | CPU                | FP32                  | 33.8ms   | 36.7ms       | 64.5ms  | 17.1ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP32                  | 9.8ms    | 10.7ms       | ×       | ?        | 3.9ms    |
| YOLOv11n           | Detect           | CPU                | FP16                  | ×        | 58.7ms       | 68.6ms  | 17.2ms   | ×        |
| YOLOv11n           | Detect           | GPU                | FP16                  | 9.9ms    | 18.1ms       | ×       | ?        | 3.7ms    |
| YOLOv11n           | Detect           | CPU                | INT8                  | ×        | 53.3ms       | ×       | 18.9ms   | ×        |
| YOLOv11n           | Detect           | GPU                | INT8                  | ×        | 43.9ms       | ×       | ?        | 3.6ms    |
| YOLOv11n           | Segment          | CPU                | FP32                  | 49.9ms   | 65.7ms       | 140.5ms | 36.1ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP32                  | 25.9ms   | 31.2ms       | ×       | ?        | 25.7ms   |
| YOLOv11n           | Segment          | CPU                | FP16                  | ×        | 92.2ms       | 149.9ms | 37.9ms   | ×        |
| YOLOv11n           | Segment          | GPU                | FP16                  | 26.1ms   | 43.8ms       | ×       | ?        | 22.5ms   |
| YOLOv11n           | Segment          | CPU                | INT8                  | ×        | 84.2ms       | ×       | 50.3ms   | ×        |
| YOLOv11n           | Segment          | GPU                | INT8                  | ×        | 79.1ms       | ×       | ?        | 23.3ms   |
| YOLOv11n           | Pose             | CPU                | FP32                  | 34.9ms   | 218.6ms      | 52.7ms  | 17.7ms   | ×        |
| YOLOv11n           | Pose             | GPU                | FP32                  | 10.3ms   | 10.3ms       | ×       | ?        | 2.7ms    |
| YOLOv11n           | Pose             | CPU                | FP16                  | ×        | 271.3ms      | 55.8ms  | 17.4ms   | ×        |
| YOLOv11n           | Pose             | GPU                | FP16                  | 9.7ms    | 12.8ms       | ×       | ?        | 2.4ms    |
| YOLOv11n           | Pose             | CPU                | INT8                  | ×        | 334.3ms      | ×       | 18.7ms   | ×        |
| YOLOv11n           | Pose             | GPU                | INT8                  | ×        | 257.2ms      | ×       | ?        | 2.3ms    |
| YOLOv11n           | OBB              | CPU                | FP32                  | 75.6ms   | 201.8ms      | 300.5ms | 175.7ms  | ×        |
| YOLOv11n           | OBB              | GPU                | FP32                  | 22.9ms   | 136.5ms      | ×       | ?        | 116.8ms  |
| YOLOv11n           | OBB              | CPU                | FP16                  | ×        | ×            | 302.9ms | 178.0ms  | ×        |
| YOLOv11n           | OBB              | GPU                | FP16                  | 22.7ms   | ×            | ×       | ?        | 148.9ms  |
| YOLOv11n           | OBB              | CPU                | INT8                  | ×        | 220.6ms      | ×       | 180.6ms  | ×        |
| YOLOv11n           | OBB              | GPU                | INT8                  | ×        | 203.2ms      | ×       | ?        | 135.3ms  |
| YOLOv12n           | Classify         | CPU                | FP32                  | 8.9ms    | 5.3ms        | 15.0ms  | 6.2ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP32                  | 2.8ms    | 2.3ms        | 5.2ms   | ?        | 1.2ms    |
| YOLOv12n           | Classify         | CPU                | FP16                  | ×        | 8.9ms        | 13.4ms  | 5.8ms    | ×        |
| YOLOv12n           | Classify         | GPU                | FP16                  | 3.0ms    | 2.3ms        | 6.1ms   | ?        | 1.2ms    |
| YOLOv12n           | Classify         | CPU                | INT8                  | ×        | 9.0ms        | ×       | 6.8ms    | ×        |
| YOLOv12n           | Classify         | GPU                | INT8                  | ×        | 8.2ms        | ×       | ?        | 1.3ms    |
| YOLOv12n           | Detect           | CPU                | FP32                  | 43.1ms   | 41.6ms       | 164.2ms | 23.0ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP32                  | 11.3ms   | 11.5ms       | 14.7ms  | ?        | 4.2ms    |
| YOLOv12n           | Detect           | CPU                | FP16                  | ×        | 68.8ms       | 167.0ms | 23.2ms   | ×        |
| YOLOv12n           | Detect           | GPU                | FP16                  | 11.9ms   | 19.7ms       | 15.2ms  | ?        | 3.9ms    |
| YOLOv12n           | Detect           | CPU                | INT8                  | ×        | 65.2ms       | ×       | 24.9ms   | ×        |
| YOLOv12n           | Detect           | GPU                | INT8                  | ×        | 51.3ms       | ×       | ?        | 4.2ms    |
| YOLOv12n           | Segment          | CPU                | FP32                  | 55.7ms   | 85.3ms       | 226.1ms | 45.6ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP32                  | 29.1ms   | 35.5ms       | 39.0ms  | ?        | 12.2ms   |
| YOLOv12n           | Segment          | CPU                | FP16                  | ×        | 111.5ms      | 217.5ms | 45.9ms   | ×        |
| YOLOv12n           | Segment          | GPU                | FP16                  | 28.9ms   | 43.9ms       | 40.4ms  | ?        | 21.3ms   |
| YOLOv12n           | Segment          | CPU                | INT8                  | ×        | 99.9ms       | ×       | 59.7ms   | ×        |
| YOLOv12n           | Segment          | GPU                | INT8                  | ×        | 98.6ms       | ×       | ?        | 23.2ms   |
| YOLOv12n           | Pose             | CPU                | FP32                  | 40.7ms   | 338.8ms      | 143.8ms | 20.8ms   | ×        |
| YOLOv12n           | Pose             | GPU                | FP32                  | 10.6ms   | 10.5ms       | 12.7ms  | ?        | 2.9ms    |
| YOLOv12n           | Pose             | CPU                | FP16                  | ×        | ×            | 144.9ms | 21.1ms   | ×        |
| YOLOv12n           | Pose             | GPU                | FP16                  | 11.3ms   | ×            | 11.3ms  | ?        | 2.7ms    |
| YOLOv12n           | Pose             | CPU                | INT8                  | ×        | 498.2ms      | ×       | 22.7ms   | ×        |
| YOLOv12n           | Pose             | GPU                | INT8                  | ×        | 392.6ms      | ×       | ?        | 2.9ms    |
| YOLOv12n           | OBB              | CPU                | FP32                  | 147.2ms  | 147.7ms      | 466.8ms | 93.1ms   | ×        |
| YOLOv12n           | OBB              | GPU                | FP32                  | 22.9ms   | 64.8ms       | 68.6ms  | ?        | 50.5ms   |
| YOLOv12n           | OBB              | CPU                | FP16                  | ×        | ×            | 479.6ms | 94.6ms   | ×        |
| YOLOv12n           | OBB              | GPU                | FP16                  | 24.3ms   | ×            | 80.0ms  | ?        | 53.9ms   |
| YOLOv12n           | OBB              | CPU                | INT8                  | ×        | 207.4ms      | ×       | 92.3ms   | ×        |
| YOLOv12n           | OBB              | GPU                | INT8                  | ×        | 161.1ms      | ×       | ?        | 50.3ms   |
| YOLOv13n           | Detect           | CPU                | FP32                  | 54.3ms   | 48.5ms       | 186.7ms | 25.5ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP32                  | 12.4ms   | 12.1ms       | 18.1ms  | ?        | 4.7ms    |
| YOLOv13n           | Detect           | CPU                | FP16                  | ×        | 99.3ms       | 184.5ms | 24.9ms   | ×        |
| YOLOv13n           | Detect           | GPU                | FP16                  | 13.3ms   | 20.1ms       | 20.1ms  | ?        | 4.3ms    |
| YOLOv13n           | Detect           | CPU                | INT8                  | ×        | 99.4ms       | ×       | 26.1ms   | ×        |
| YOLOv13n           | Detect           | GPU                | INT8                  | ×        | 85.1ms       | ×       | ?        | 5.1ms    |
| YOLO26n            | Classify         | CPU                | FP32                  | 4.9ms    | 3.2ms        | 5.5ms   | 4.0ms    | ×        |
| YOLO26n            | Classify         | GPU                | FP32                  | 1.8ms    | 4.1ms        | ×       | ?        | 0.8ms    |
| YOLO26n            | Classify         | CPU                | FP16                  | ×        | 5.0ms        | 5.8ms   | 4.0ms    | ×        |
| YOLO26n            | Classify         | GPU                | FP16                  | 1.9ms    | 6.1ms        | ×       | ?        | 0.7ms    |
| YOLO26n            | Classify         | CPU                | INT8                  | ×        | 5.0ms        | ×       | 4.2ms   | ×        |
| YOLO26n            | Classify         | GPU                | INT8                  | ×        | 5.3ms        | ×       | ?        | 0.6ms    |
| YOLO26n            | Detect           | CPU                | FP32                  | 31.1ms   | 29.5ms       | 60.1ms  | 16.8ms   | ×        |
| YOLO26n            | Detect           | GPU                | FP32                  | 9.8ms    | 36.8ms       | ×       | ?        | 2.1ms    |
| YOLO26n            | Detect           | CPU                | FP16                  | ×        | ×            | 61.2ms  | 16.7ms   | ×        |
| YOLO26n            | Detect           | GPU                | FP16                  | 10.0ms   | ×            | ×       | ?        | 1.7ms    |
| YOLO26n            | Detect           | CPU                | INT8                  | ×        | 49.8ms       | ×       | 17.4ms   | ×        |
| YOLO26n            | Detect           | GPU                | INT8                  | ×        | 47.5ms       | ×       | ?        | 1.7ms    |
| YOLO26n            | Segment          | CPU                | FP32                  | 45.3ms   | 65.2ms       | 84.6ms  | 30.6ms   | ×        |
| YOLO26n            | Segment          | GPU                | FP32                  | 23.1ms   | 70.3ms       | ×       | ?        | 22.0ms   |
| YOLO26n            | Segment          | CPU                | FP16                  | ×        | ×            | 91.0ms  | 30.9ms   | ×        |
| YOLO26n            | Segment          | GPU                | FP16                  | 23.6ms   | ×            | ×       | ?        | 21.3ms   |
| YOLO26n            | Segment          | CPU                | INT8                  | ×        | 83.7ms       | ×       | 40.3ms   | ×        |
| YOLO26n            | Segment          | GPU                | INT8                  | ×        | 90.7ms       | ×       | ?        | 21.2ms   |
| YOLO26n            | Pose             | CPU                | FP32                  | 38.1ms   | 32.7ms       | ×       | 17.7ms   | ×        |
| YOLO26n            | Pose             | GPU                | FP32                  | 10.6ms   | 38.7ms       | ×       | ?        | 2.3ms    |
| YOLO26n            | Pose             | CPU                | FP16                  | ×        | ×            | ×       | 17.3ms   | ×        |
| YOLO26n            | Pose             | GPU                | FP16                  | 10.6ms   | ×            | ×       | ?        | 1.9ms    |
| YOLO26n            | Pose             | CPU                | INT8                  | ×        | 53.0ms       | ×       | 18.1ms   | ×        |
| YOLO26n            | Pose             | GPU                | INT8                  | ×        | 51.2ms       | ×       | ?        | 1.7ms    |
| YOLO26n            | OBB              | CPU                | FP32                  | 63.4ms   | 70.6ms       | 168.9ms | 27.3ms   | ×        |
| YOLO26n            | OBB              | GPU                | FP32                  | 21.8ms   | 78.2ms       | ×       | ?        | 3.7ms    |
| YOLO26n            | OBB              | CPU                | FP16                  | ×        | ×            | 172.8ms | 27.2ms   | ×        |
| YOLO26n            | OBB              | GPU                | FP16                  | 21.4ms   | ×            | ×       | ?        | 3.1ms    |
| YOLO26n            | OBB              | CPU                | INT8                  | ×        | 105.9ms      | ×       | 29.0ms   | ×        |
| YOLO26n            | OBB              | GPU                | INT8                  | ×        | 107.9ms      | ×       | ?        | 3.2ms    |


You can download some model weights in: <https://pan.baidu.com/s/1843WW7tNQK1ycqIALje_fA?pwd=adis>

**For your own model, you should transpose output dims such as from 1x84x8400 to 1x8400x84 for YOLOv8, YOLOv9, YOLOv11, YOLOv12, YOLOv13 detection, segmentation, pose and obb.**