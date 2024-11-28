# yolo-inference
C++ and Python implementations of YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11 inference.

Supported inference backends include Libtorch/PyTorch, ONNXRuntime, OpenCV, OpenVINO and TensorRT. 

Supported task types include Classify, Detect and Segment.

Supported model types include FP32, FP16 and INT8.

Dependencies:
* [CUDA](https://developer.nvidia.com/cuda-downloads) version 11.8.0/12.5.0
* [OpenCV](https://github.com/opencv/opencv) version 4.9.0/4.10.0 (built with CUDA)
* [ONNXRuntime](https://github.com/microsoft/onnxruntime) version 1.18.1/1.20.0
* [OpenVINO](https://github.com/openvinotoolkit/openvino) version 2024.1.0/2024.4.0
* [TensorRT](https://developer.nvidia.com/tensorrt/download) version 8.2.1.8/10.6.0.26
* [Torch](https://pytorch.org) version 2.0.0+cu118/2.5.0+cu124

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

C++ test in Docker(CPU i7-13700F, GPU RTX4070): 
|       Model       |       Task       |       Device       |       Precision       | LibTorch | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :------: | :---------: | :----: | :------: | :------: |
| YOLOv5n | Classify | CPU | FP32 | 11.0ms | 12.0ms | 14.0ms | 9.8ms | × |
| YOLOv5n | Classify | GPU | FP32 | 3.2ms | 5.6ms | 4.1ms | ? | 2.6ms |
| YOLOv5n | Classify | CPU | FP16 | × | 17.6ms | 14.2ms | 9.8ms | × |
| YOLOv5n | Classify | GPU | FP16 | 3.7ms | 7.2ms | 4.0ms | ? | 2.4ms |
| YOLOv5n | Classify | CPU | INT8 | × | 16.8ms | × | ? | × |
| YOLOv5n | Classify | GPU | INT8 | × | 30.5ms | × | ? | 2.4ms |
| YOLOv5n | Detect | CPU | FP32 | 20.9ms | 16.5ms | 45.2ms | 14.1ms | × |
| YOLOv5n | Detect | GPU | FP32 | 4.1ms | 7.1ms | 6.0ms | ? | 2.9ms |
| YOLOv5n | Detect | CPU | FP16 | × | 31.7ms | 45.3ms | 14.1ms | × |
| YOLOv5n | Detect | GPU | FP16 | 3.7ms | 16.5ms | 5.8ms | ? | 2.5ms |
| YOLOv5n | Detect | CPU | INT8 | × | 22.5ms | × | 13.7ms | × |
| YOLOv5n | Detect | GPU | INT8 | × | 44.7ms | × | ? | 2.5ms |
| YOLOv5n | Segment | CPU | FP32 | 27.5ms | 22.9ms | 61.8ms | 20.2ms | × |
| YOLOv5n | Segment | GPU | FP32 | 6.9ms | 10.7ms | 7.8ms | ? | 4.5ms |
| YOLOv5n | Segment | CPU | FP16 | × | 43.7ms | 61.5ms | 20.2ms | × |
| YOLOv5n | Segment | GPU | FP16 | 6.3ms | 27.8ms | 7.4ms | ? | 4.0ms |
| YOLOv5n | Segment | CPU | INT8 | × | 30.4ms | × | ? | × |
| YOLOv5n | Segment | GPU | INT8 | × | 62.7ms | × | ? | ? |
| YOLOv6n | Detect | CPU | FP32 | ? | 19.7ms | 21.8ms | 21.7ms | × |
| YOLOv6n | Detect | GPU | FP32 | ? | 7.3ms | 4.9ms | ? | 3.3ms |
| YOLOv6n | Detect | CPU | FP16 | × | 37.7ms | 21.7ms | 21.8ms | × |
| YOLOv6n | Detect | GPU | FP16 | ? | 14.5ms | 4.5ms | ? | 2.6ms |
| YOLOv6n | Detect | CPU | INT8 | × | 34.1ms | × | 18.9ms | × |
| YOLOv6n | Detect | GPU | INT8 | × | 64.6ms | × | ? | 2.4ms |
| YOLOv7t | Detect | CPU | FP32 | 42.9ms | 24.6ms | 49.5ms | 27.6ms | × |
| YOLOv7t | Detect | GPU | FP32 | 4.9ms | 7.8ms | 6.5ms | ? | 3.5ms |
| YOLOv7t | Detect | CPU | FP16 | × | 54.2ms | 49.4ms | 27.5ms | × |
| YOLOv7t | Detect | GPU | FP16 | ? | 22.6ms | 5.7ms | ? | 2.8ms |
| YOLOv7t | Detect | CPU | INT8 | × | 42.4ms | × | 24.3ms | × |
| YOLOv7t | Detect | GPU | INT8 | × | 77.4ms | × | ? | 2.5ms |
| YOLOv8n | Classify | CPU | FP32 | 3.0ms | 1.7ms | 2.5ms | 1.5ms | × |
| YOLOv8n | Classify | GPU | FP32 | 0.9ms | 1.0ms | 1.4ms | ? | 0.7ms |
| YOLOv8n | Classify | CPU | FP16 | × | 3.3ms | 2.5ms | 1.5ms | × |
| YOLOv8n | Classify | GPU | FP16 | ? | 1.3ms | 1.5ms | ? | 0.6ms |
| YOLOv8n | Classify | CPU | INT8 | × | 2.7ms | × | ? | × |
| YOLOv8n | Classify | GPU | INT8 | × | 5.9ms | × | ? | 0.6ms |
| YOLOv8n | Detect | CPU | FP32 | 26.3ms | 24.8ms | 29.0ms | 21.2ms | × |
| YOLOv8n | Detect | GPU | FP32 | 3.8ms | 7.7ms | 5.1ms | ? | 3.3ms |
| YOLOv8n | Detect | CPU | FP16 | × | 42.7ms | 28.9ms | 21.3ms | × |
| YOLOv8n | Detect | GPU | FP16 | ? | 20.0ms | 4.7ms | ? | 2.8ms |
| YOLOv8n | Detect | CPU | INT8 | × | 32.6ms | × | 19.3ms | × |
| YOLOv8n | Detect | GPU | INT8 | × | 59.7ms | × | ? | 2.6ms |
| YOLOv8n | Segment | CPU | FP32 | ? | 34.0ms | 38.1ms | 28.4ms | × |
| YOLOv8n | Segment | GPU | FP32 | 6.3ms | 10.5ms | 6.8ms | ? | 4.9ms |
| YOLOv8n | Segment | CPU | FP16 | × | 55.8ms | 37.9ms | 28.4ms | × |
| YOLOv8n | Segment | GPU | FP16 | ? | 26.9ms | 6.5ms | ? | 4.3ms |
| YOLOv8n | Segment | CPU | INT8 | × | 42.8ms | × | ? | × |
| YOLOv8n | Segment | GPU | INT8 | × | 81.3ms | × | ? | ? |
| YOLOv9t | Detect | CPU | FP32 | 36.6ms | 27.0ms | 35.4ms | 21.8ms | × |
| YOLOv9t | Detect | GPU | FP32 | 6.1ms | 9.8m | 8.4ms | ? | 4.3ms |
| YOLOv9t | Detect | CPU | FP16 | × | 44.0ms | 35.5ms | 21.8ms | × |
| YOLOv9t | Detect | GPU | FP16 | ? | 19.9ms | 8.7ms | ? | 3.7ms |
| YOLOv9t | Detect | CPU | INT8 | × | 39.7ms | × | 20.4ms | × |
| YOLOv9t | Detect | GPU | INT8 | × | 88.3ms | × | ? | 3.6ms |
| YOLOv10n | Detect | CPU | FP32 | 25.3ms | 23.3ms | x | 18.6ms | × |
| YOLOv10n | Detect | GPU | FP32 | 3.0ms | 7.3m | × | ? | 3.0ms |
| YOLOv10n | Detect | CPU | FP16 | × | 41.6ms | × | 18.6ms | × |
| YOLOv10n | Detect | GPU | FP16 | ? | 12.0ms | × | ? | 2.4ms |
| YOLOv10n | Detect | CPU | INT8 | × | 35.0ms | × | 17.3ms | × |
| YOLOv10n | Detect | GPU | INT8 | × | 63.7ms | × | ? | 2.3ms |
| YOLOv11n | Classify | CPU | FP32 | 2.9ms | 1.9ms | 2.9ms | 1.6ms | × |
| YOLOv11n | Classify | GPU | FP32 | 1.2ms | 1.3ms | × | ? | 0.9ms |
| YOLOv11n | Classify | CPU | FP16 | × | 3.4ms | 3.0ms | 1.7ms | × |
| YOLOv11n | Classify | GPU | FP16 | ? | 1.6ms | × | ? | 0.7ms |
| YOLOv11n | Classify | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Classify | GPU | INT8 | × | ? | × | ? | 0.7ms |
| YOLOv11n | Detect | CPU | FP32 | 28.9ms | 23.3ms | 31.4ms | 18.5ms | × |
| YOLOv11n | Detect | GPU | FP32 | 4.1ms | 8.2ms | × | ? | 3.4ms |
| YOLOv11n | Detect | CPU | FP16 | × | 46.4ms | 31.1ms | 18.4ms | × |
| YOLOv11n | Detect | GPU | FP16 | ? | 17.6ms | × | ? | 3.0ms |
| YOLOv11n | Detect | CPU | INT8 | × | ? | × | 17.0ms | × |
| YOLOv11n | Detect | GPU | INT8 | × | ? | × | ? | 2.9ms |
| YOLOv11n | Segment | CPU | FP32 | × | 32.3ms | 40.5ms | 25.7ms | × |
| YOLOv11n | Segment | GPU | FP32 | × | 11.8ms | × | ? | 5.0ms |
| YOLOv11n | Segment | CPU | FP16 | × | 59.8ms | 40.1ms | 25.6ms | × |
| YOLOv11n | Segment | GPU | FP16 | × | 27.1ms | × | ? | 4.5ms |
| YOLOv11n | Segment | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Segment | GPU | INT8 | × | ? | × | ? | ? |

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

Python test in Docker(CPU i7-13700F, GPU RTX4070): 
|       Model       |       Task       |       Device       |       Precision       | PyTorch | ONNXRuntime | OpenCV | OpenVINO | TensorRT |
| :---------------: | :--------------: | :----------------: | :-------------------: | :-----: | :---------: | :----: | :------: | :------: |
| YOLOv5n | Classify | CPU | FP32 | 19.1ms | 19.3ms | 23.3ms | 17.6ms | × |
| YOLOv5n | Classify | GPU | FP32 | 11.8ms | 15.1ms | 12.6ms | ? | 8.9ms |
| YOLOv5n | Classify | CPU | FP16 | × | 24.9ms | 23.5ms | 17.6ms | × |
| YOLOv5n | Classify | GPU | FP16 | 13.5ms | 15.4ms | 13.2ms | ? | 10.2ms |
| YOLOv5n | Classify | CPU | INT8 | × | 25.2ms | × | ? | × |
| YOLOv5n | Classify | GPU | INT8 | × | 39.2ms | × | ? | 10.3ms |
| YOLOv5n | Detect | CPU | FP32 | 22.3ms | 21.0ms | 47.7ms | 18.2ms | × |
| YOLOv5n | Detect | GPU | FP32 | 9.1ms | 12.8ms | 8.1ms | ? | 5.5ms |
| YOLOv5n | Detect | CPU | FP16 | × | 32.6ms | 46.9ms | 18.2ms | × |
| YOLOv5n | Detect | GPU | FP16 | 8.3ms | 15.4ms | 8.0ms | ? | 5.1ms |
| YOLOv5n | Detect | CPU | INT8 | × | 27.9ms | × | 17.9ms | × |
| YOLOv5n | Detect | GPU | INT8 | × | 54.7ms | × | ? | 6.0ms |
| YOLOv5n | Segment | CPU | FP32 | 154.8ms | 98.1ms | 129.0ms | 44.3ms | × |
| YOLOv5n | Segment | GPU | FP32 | 31.2ms | 42.6ms | 31.4ms | ? | 32.6ms |
| YOLOv5n | Segment | CPU | FP16 | × | 119.0ms | 129.3ms | 44.7ms | × |
| YOLOv5n | Segment | GPU | FP16 | 42.3ms | 50.2ms | 31.6ms | ? | 33.4ms |
| YOLOv5n | Segment | CPU | INT8 | × | 112.5ms | × | ? | × |
| YOLOv5n | Segment | GPU | INT8 | × | 166.6ms | × | ? | ? |
| YOLOv6n | Detect | CPU | FP32 | ? | 45.8ms | 37.5ms | 39.3ms | × |
| YOLOv6n | Detect | GPU | FP32 | ? | 36.6ms | 30.6ms | ? | 27.8ms |
| YOLOv6n | Detect | CPU | FP16 | × | 58.6ms | 39.2ms | 39.4ms | × |
| YOLOv6n | Detect | GPU | FP16 | ? | 35.3ms | 29.1ms | ? | 24.0ms |
| YOLOv6n | Detect | CPU | INT8 | × | 59.4ms | × | 36.5ms | × |
| YOLOv6n | Detect | GPU | INT8 | × | 110.8ms | × | ? | 22.1ms |
| YOLOv7t | Detect | CPU | FP32 | 47.0ms | 32.3ms | 52.0s | 31.8ms | × |
| YOLOv7t | Detect | GPU | FP32 | 8.0ms | 12.6ms | 8.9ms | ? | 6.1ms |
| YOLOv7t | Detect | CPU | FP16 | × | 55.6ms | 52.2ms | 31.7ms | × |
| YOLOv7t | Detect | GPU | FP16 | ? | 18.9ms | 7.8ms | ? | 5.4ms |
| YOLOv7t | Detect | CPU | INT8 | × | 48.6ms | × | 27.5ms | × |
| YOLOv7t | Detect | GPU | INT8 | × | 90.9ms | × | ? | 5.0ms |
| YOLOv8n | Classify | CPU | FP32 | 3.3ms | 1.9ms | 2.7ms | 1.4ms | × |
| YOLOv8n | Classify | GPU | FP32 | 1.1ms | 1.1ms | 1.6ms | ? | 0.7ms |
| YOLOv8n | Classify | CPU | FP16 | × | 3.6ms | 2.6ms | 1.4ms | × |
| YOLOv8n | Classify | GPU | FP16 | ? | 1.4ms | 1.6ms | ? | 0.6ms |
| YOLOv8n | Classify | CPU | INT8 | × | 3.3ms | × | ? | × |
| YOLOv8n | Classify | GPU | INT8 | × | 6.5ms | × | ? | 0.6ms |
| YOLOv8n | Detect | CPU | FP32 | 45.2ms | 53.7ms | 45.4ms | 37.3ms | × |
| YOLOv8n | Detect | GPU | FP32 | 28.5ms | 33.9ms | 28.3ms | ? | 25.9ms |
| YOLOv8n | Detect | CPU | FP16 | × | 61.7ms | 43.8ms | 37.4ms | × |
| YOLOv8n | Detect | GPU | FP16 | ? | 39.5ms | 27.5ms | ? | 22.9ms |
| YOLOv8n | Detect | CPU | INT8 | × | 59.2ms | × | 35.5ms | × |
| YOLOv8n | Detect | GPU | INT8 | × | 93.3ms | × | ? | 21.6ms |
| YOLOv8n | Segment | CPU | FP32 | 170.8ms | 144.0ms | 133.8ms | 87.9ms | × |
| YOLOv8n | Segment | GPU | FP32 | 78.1ms | 84.1ms | 81.5ms | ? | 70.1ms |
| YOLOv8n | Segment | CPU | FP16 | × | 155.6ms | 132.9ms | 89.1ms | × |
| YOLOv8n | Segment | GPU | FP16 | ? | 85.8ms | 76.8ms | ? | 76.6ms |
| YOLOv8n | Segment | CPU | INT8 | × | 157.0ms | × | ? | × |
| YOLOv8n | Segment | GPU | INT8 | × | 210.2ms | × | ? | ? |
| YOLOv9t | Detect | CPU | FP32 | 57.7ms | 52.3ms | 52.5ms | 37.9ms | × |
| YOLOv9t | Detect | GPU | FP32 | 30.4ms | 39.1ms | 32.6ms | ? | 27.6ms |
| YOLOv9t | Detect | CPU | FP16 | × | 67.3ms | 51.9ms | 38.1ms | × |
| YOLOv9t | Detect | GPU | FP16 | ? | 42.0ms | 31.9ms | ? | 26.9ms |
| YOLOv9t | Detect | CPU | INT8 | × | 67.5ms | × | 36.5ms | × |
| YOLOv9t | Detect | GPU | INT8 | × | 122.3ms | × | ? | 26.2ms |
| YOLOv10n | Detect | CPU | FP32 | 29.5ms | 32.8ms | × | 20.6ms | × |
| YOLOv10n | Detect | GPU | FP32 | 5.5ms | 11.2m | × | ? | 5.3ms |
| YOLOv10n | Detect | CPU | FP16 | × | 44.2ms | × | 20.6ms | × |
| YOLOv10n | Detect | GPU | FP16 | ? | 12.7ms | × | ? | 4.7ms |
| YOLOv10n | Detect | CPU | INT8 | × | 47.8ms | × | 19.5ms | × |
| YOLOv10n | Detect | GPU | INT8 | × | 78.0ms | × | ? | 4.6ms |
| YOLOv11n | Classify | CPU | FP32 | 3.4ms | 2.2ms | 3.0ms | 1.5ms | × |
| YOLOv11n | Classify | GPU | FP32 | 1.4ms | 1.4ms | × | ? | 0.8ms |
| YOLOv11n | Classify | CPU | FP16 | × | 4.0ms | 3.0ms | 1.6ms | × |
| YOLOv11n | Classify | GPU | FP16 | ? | 1.6ms | × | ? | 0.7ms |
| YOLOv11n | Classify | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Classify | GPU | INT8 | × | ? | × | ? | 0.7ms |
| YOLOv11n | Detect | CPU | FP32 | 48.1ms | 49.1ms | 46.5ms | 34.7ms | × |
| YOLOv11n | Detect | GPU | FP32 | 30.5ms | 34.2ms | × | ? | 26.9ms |
| YOLOv11n | Detect | CPU | FP16 | × | 69.2ms | 46.7ms | 34.7ms | × |
| YOLOv11n | Detect | GPU | FP16 | ? | 36.9ms | × | ? | 23.8ms |
| YOLOv11n | Detect | CPU | INT8 | × | ? | × | 33.8ms | × |
| YOLOv11n | Detect | GPU | INT8 | × | ? | × | ? | 22.4ms |
| YOLOv11n | Segment | CPU | FP32 | 171.1ms | 137.4ms | 137.8ms | 81.6ms | × |
| YOLOv11n | Segment | GPU | FP32 | 79.1ms | 81.9ms | × | ? | 64.6ms |
| YOLOv11n | Segment | CPU | FP16 | × | 159.2ms | 137.3ms | 80.8ms | × |
| YOLOv11n | Segment | GPU | FP16 | ?ms | 84.5ms | × | ? | 58.6ms |
| YOLOv11n | Segment | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Segment | GPU | INT8 | × | ? | × | ? | ? |

You can get a docker image with:
```bash
docker pull taify/yolo_inference:cuda11.8
```
or
```bash
docker pull taify/yolo_inference:cuda12.5
```

You Can download some model weights in:  <https://pan.baidu.com/s/1L8EyTa59qu_eEb3lKRnPQA?pwd=itda>


For your own model, you should convert onnx model with following scirpt to transpose output dims for YOLOv8, YOLOv9, YOLOv11 detection and segmentation:
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