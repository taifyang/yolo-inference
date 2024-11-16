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
| YOLOv5n | Classify | CPU | FP32 | 15.3ms | 12.2ms | 20.6ms | 14.1ms | × |
| YOLOv5n | Classify | GPU | FP32 | 4.9ms | 5.1ms | 5.1ms | ? | 4.1ms |
| YOLOv5n | Classify | CPU | FP16 | × | 21.7ms | 20.1ms | 14.0ms | × |
| YOLOv5n | Classify | GPU | FP16 | 4.6ms | 8.1ms | 4.9ms | ? | 3.2ms |
| YOLOv5n | Classify | CPU | INT8 | × | 18.3ms | × | ? | × |
| YOLOv5n | Classify | GPU | INT8 | × | 34.2ms | × | ? | 3.0ms |
| YOLOv5n | Detect | CPU | FP32 | 23.3ms | 20.2ms | 57.3ms | 20.0ms | × |
| YOLOv5n | Detect | GPU | FP32 | 7.2ms | 6.4ms | 8.2ms | ? | 4.4ms |
| YOLOv5n | Detect | CPU | FP16 | × | 41.8ms | 57.3ms | 19.8ms | × |
| YOLOv5n | Detect | GPU | FP16 | 6.8ms | 18.8ms | 7.9ms | ? | 3.9ms |
| YOLOv5n | Detect | CPU | INT8 | × | 26.7ms | × | 18.1ms | × |
| YOLOv5n | Detect | GPU | INT8 | × | 49.3ms | × | ? | 3.5ms |
| YOLOv5n | Segment | CPU | FP32 | × | 28.2ms | 75.8ms | 27.2ms | × |
| YOLOv5n | Segment | GPU | FP32 | 10.6ms | 10.6ms | 10.8ms | ? | 6.3ms |
| YOLOv5n | Segment | CPU | FP16 | × | 55.0ms | 75.9ms | 27.2ms | × |
| YOLOv5n | Segment | GPU | FP16 | 9.8ms | 29.0ms | 10.0ms | ? | 5.0ms |
| YOLOv5n | Segment | CPU | INT8 | × | 34.5ms | × | ? | × |
| YOLOv5n | Segment | GPU | INT8 | × | 62.1ms | × | ? | 4.2ms |
| YOLOv6n | Detect | CPU | FP32 | ? | 28.1ms | 29.7ms | 29.3ms | × |
| YOLOv6n | Detect | GPU | FP32 | ? | 6.4ms | 6.5ms | ? | 5.0ms |
| YOLOv6n | Detect | CPU | FP16 | × | 47.1ms | 27.4ms | 29.3ms | × |
| YOLOv6n | Detect | GPU | FP16 | ? | 13.1ms | 6.2ms | ? | 3.5ms |
| YOLOv6n | Detect | CPU | INT8 | × | 38.5ms | × | 23.4ms | × |
| YOLOv6n | Detect | GPU | INT8 | × | 95.7ms | × | ? | 4.1ms |
| YOLOv7t | Detect | CPU | FP32 | 50.5ms | 33.6ms | 59.9ms | 34.8ms | × |
| YOLOv7t | Detect | GPU | FP32 | 8.0ms | 7.7ms | 8.7ms | ? | 5.5ms |
| YOLOv7t | Detect | CPU | FP16 | × | 71.7ms | 63.7ms | 34.7ms | × |
| YOLOv7t | Detect | GPU | FP16 | ? | 21.3ms | 7.0ms | ? | 3.9ms |
| YOLOv7t | Detect | CPU | INT8 | × | 50.7ms | × | 27.8ms | × |
| YOLOv7t | Detect | GPU | INT8 | × | 85.6ms | × | ? | 3.7ms |
| YOLOv8n | Classify | CPU | FP32 | 3.5ms | 2.2ms | 4.0ms | 2.4ms | × |
| YOLOv8n | Classify | GPU | FP32 | 2.3ms | 1.5ms | 1.9ms | ? | 1.2ms |
| YOLOv8n | Classify | CPU | FP16 | × | 6.3ms | 4.0ms | 2.4ms | × |
| YOLOv8n | Classify | GPU | FP16 | ? | 1.7ms | 1.7ms | ? | 1.0ms |
| YOLOv8n | Classify | CPU | INT8 | × | 3.4ms | × | ? | × |
| YOLOv8n | Classify | GPU | INT8 | × | 7.8ms | × | ? | 1.0ms |
| YOLOv8n | Detect | CPU | FP32 | 33.3ms | 27.9ms | 42.2ms | 28.6ms | × |
| YOLOv8n | Detect | GPU | FP32 | 6.4ms | 6.9ms | 6.8ms | ? | 6.0ms |
| YOLOv8n | Detect | CPU | FP16 | × | 57.2ms | 41.9ms | 28.6ms | × |
| YOLOv8n | Detect | GPU | FP16 | ? | 19.4ms | 5.7ms | ? | 3.7ms |
| YOLOv8n | Detect | CPU | INT8 | × | 37.3ms | × | 24.5ms | × |
| YOLOv8n | Detect | GPU | INT8 | × | 85.5ms | × | ? | 4.7ms |
| YOLOv8n | Segment | CPU | FP32 | × | 42.9ms | 54.7ms | 37.5ms | × |
| YOLOv8n | Segment | GPU | FP32 | 9.5ms | 10.5ms | × | ? | 8.1ms |
| YOLOv8n | Segment | CPU | FP16 | × | 73.1ms | 54.9ms | 37.4ms | × |
| YOLOv8n | Segment | GPU | FP16 | ? | 27.3ms | × | ? | 5.9ms |
| YOLOv8n | Segment | CPU | INT8 | × | 51.0ms | × | ? | × |
| YOLOv8n | Segment | GPU | INT8 | × | 101.1ms | × | ? | 5.6ms |
| YOLOv9t | Detect | CPU | FP32 | 40.8ms | 34.6ms | 54.1ms | 29.0ms | × |
| YOLOv9t | Detect | GPU | FP32 | 8.1ms | 9.4m | 9.7ms | ? | 7.1ms |
| YOLOv9t | Detect | CPU | FP16 | × | 60.6ms | 55.0ms | 29.0ms | × |
| YOLOv9t | Detect | GPU | FP16 | ? | 17.9ms | 9.0ms | ? | 4.9ms |
| YOLOv9t | Detect | CPU | INT8 | × | 48.0ms | × | 27.0ms | × |
| YOLOv9t | Detect | GPU | INT8 | × | 135.2ms | × | ? | 5.6ms |
| YOLOv10n | Detect | CPU | FP32 | 30.4ms | 27.9ms | × | 26.1ms | × |
| YOLOv10n | Detect | GPU | FP32 | 6.0ms | 6.5m | × | ? | × |
| YOLOv10n | Detect | CPU | FP16 | × | 56.4ms | × | 26.0ms | × |
| YOLOv10n | Detect | GPU | FP16 | ? | 10.9ms | × | ? | × |
| YOLOv10n | Detect | CPU | INT8 | × | 40.7ms | × | 23.5ms | × |
| YOLOv10n | Detect | GPU | INT8 | × | 83.9ms | × | ? | × |
| YOLOv11n | Classify | CPU | FP32 | 4.1ms | 2.4ms | 4.4ms | 2.6ms | × |
| YOLOv11n | Classify | GPU | FP32 | 2.7ms | 1.7ms | × | ? | 1.4ms |
| YOLOv11n | Classify | CPU | FP16 | × | 6.3ms | 4.5ms | 2.6ms | × |
| YOLOv11n | Classify | GPU | FP16 | ? | 2.1ms | × | ? | 1.1ms |
| YOLOv11n | Classify | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Classify | GPU | INT8 | × | ? | × | ? | 1.3ms |
| YOLOv11n | Detect | CPU | FP32 | 35.0ms | 26.9ms | 44.4ms | 25.0ms | × |
| YOLOv11n | Detect | GPU | FP32 | 7.2ms | 7.2ms | × | ? | 6.0ms |
| YOLOv11n | Detect | CPU | FP16 | × | 61.3ms | 44.8ms | 25.0ms | × |
| YOLOv11n | Detect | GPU | FP16 | ? | 20.0ms | × | ? | 3.9ms |
| YOLOv11n | Detect | CPU | INT8 | × | ? | × | 22.8ms | × |
| YOLOv11n | Detect | GPU | INT8 | × | ? | × | ? | 4.7ms |
| YOLOv11n | Segment | CPU | FP32 | × | 38.8ms | 56.9ms | 34.0ms | × |
| YOLOv11n | Segment | GPU | FP32 | × | 10.9ms | × | ? | 7.5ms |
| YOLOv11n | Segment | CPU | FP16 | × | 78.3ms | 58.1ms | 33.8ms | × |
| YOLOv11n | Segment | GPU | FP16 | × | 27.9ms | × | ? | 6.2ms |
| YOLOv11n | Segment | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Segment | GPU | INT8 | × | ? | × | ? | 4.9ms |

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
| YOLOv5n | Classify | CPU | FP32 | 26.3ms | 21.4ms | 33.2ms | 21.8ms | × |
| YOLOv5n | Classify | GPU | FP32 | 15.6ms | 16.1ms | 16.6ms | ? | 17.0ms |
| YOLOv5n | Classify | CPU | FP16 | × | 30.3ms | 31.5ms | 21.7ms | × |
| YOLOv5n | Classify | GPU | FP16 | 14.5ms | 18.6ms | 17.4ms | ? | 19.8ms |
| YOLOv5n | Classify | CPU | INT8 | × | 28.9ms | × | ? | × |
| YOLOv5n | Classify | GPU | INT8 | × | 54.8ms | × | ? | 18.9ms |
| YOLOv5n | Detect | CPU | FP32 | 30.6ms | 27.0ms | 60.0ms | 24.8ms | × |
| YOLOv5n | Detect | GPU | FP32 | 10.4ms | 14.9ms | 10.7ms | ? | 14.3ms |
| YOLOv5n | Detect | CPU | FP16 | × | 40.7ms | 59.8ms | 24.8ms | × |
| YOLOv5n | Detect | GPU | FP16 | 12.3ms | 19.6ms | 10.3ms | ? | 12.8ms |
| YOLOv5n | Detect | CPU | INT8 | × | 33.7ms | × | 23.1ms | × |
| YOLOv5n | Detect | GPU | INT8 | × | 72.9ms | × | ? | 13.8ms |
| YOLOv5n | Segment | CPU | FP32 | 159.2ms | 116.1ms | 147.2ms | 47.8ms | × |
| YOLOv5n | Segment | GPU | FP32 | 34.6ms | 49.1ms | 38.0ms | ? | 70.7ms |
| YOLOv5n | Segment | CPU | FP16 | × | 138.8ms | 142.2ms | 48.2ms | × |
| YOLOv5n | Segment | GPU | FP16 | 50.9ms | 78.9ms | 52.4ms | ? | 72.6ms |
| YOLOv5n | Segment | CPU | INT8 | × | 127.6ms | × | ? | × |
| YOLOv5n | Segment | GPU | INT8 | × | 191.8ms | × | ? | 13.3ms |
| YOLOv6n | Detect | CPU | FP32 | ? | 54.0ms | 48.1ms | 52.0ms | × |
| YOLOv6n | Detect | GPU | FP32 | ? | 40.0ms | 34.2ms | ? | 43.0ms |
| YOLOv6n | Detect | CPU | FP16 | × | 66.4ms | 48.1ms | 51.8ms | × |
| YOLOv6n | Detect | GPU | FP16 | ? | 49.9ms | 36.3ms | ? | 40.5ms |
| YOLOv6n | Detect | CPU | INT8 | × | 67.1ms | × | 44.9ms | × |
| YOLOv6n | Detect | GPU | INT8 | × | 241.4ms | × | ? | 61.7ms |
| YOLOv7t | Detect | CPU | FP32 | 53.3ms | 41.1ms | 62.9ms | 39.4ms | × |
| YOLOv7t | Detect | GPU | FP32 | 10.6ms | 16.5ms | 10.4ms | ? | 14.0ms |
| YOLOv7t | Detect | CPU | FP16 | × | 72.2ms | 62.9ms | 39.4ms | × |
| YOLOv7t | Detect | GPU | FP16 | ? | 24.3ms | 9.1ms | ? | 12.7ms |
| YOLOv7t | Detect | CPU | INT8 | × | 58.2ms | × | 32.4ms | × |
| YOLOv7t | Detect | GPU | INT8 | × | 101.8ms | × | ? | 12.9ms |
| YOLOv8n | Classify | CPU | FP32 | 3.5ms | 2.2ms | 4.1ms | 2.3ms | × |
| YOLOv8n | Classify | GPU | FP32 | 2.5ms | 1.6ms | 1.8ms | ? | 3.5ms |
| YOLOv8n | Classify | CPU | FP16 | × | 6.3ms | 4.1s | 2.3ms | × |
| YOLOv8n | Classify | GPU | FP16 | ? | 1.7ms | 1.7ms | ? | 2.8ms |
| YOLOv8n | Classify | CPU | INT8 | × | 3.7ms | × | ? | × |
| YOLOv8n | Classify | GPU | INT8 | × | 8.2ms | × | ? | 3.0ms |
| YOLOv8n | Detect | CPU | FP32 | 59.2ms | 57.8ms | 60.3s | 49.4ms | × |
| YOLOv8n | Detect | GPU | FP32 | 35.5ms | 40.5ms | 29.4ms | ? | 39.1ms |
| YOLOv8n | Detect | CPU | FP16 | × | 77.1ms | 61.3ms | 49.6ms | × |
| YOLOv8n | Detect | GPU | FP16 | ? | 60.4ms | 30.8ms | ? | 38.1ms |
| YOLOv8n | Detect | CPU | INT8 | × | 64.1ms | × | 44.1ms | × |
| YOLOv8n | Detect | GPU | INT8 | × | 138.7ms | × | ? | 40.9ms |
| YOLOv8n | Segment | CPU | FP32 | 184.7ms | 157.8ms | 142.3ms | 100.0ms | × |
| YOLOv8n | Segment | GPU | FP32 | 94.3ms | 104.2ms | 88.5ms | ? | 116.6ms |
| YOLOv8n | Segment | CPU | FP16 | × | 180.4ms | 144.8s | 99.3ms | × |
| YOLOv8n | Segment | GPU | FP16 | ? | 122.2ms | 108.7ms | ? | 118.7ms |
| YOLOv8n | Segment | CPU | INT8 | × | 166.4ms | × | ? | × |
| YOLOv8n | Segment | GPU | INT8 | × | 275.3ms | × | ? | 40.9ms |
| YOLOv9t | Detect | CPU | FP32 | 61.0ms | 61.0ms | 74.9ms | 49.7ms | × |
| YOLOv9t | Detect | GPU | FP32 | 33.6ms | 41.4m | 31.2ms | ? | 40.2ms |
| YOLOv9t | Detect | CPU | FP16 | × | 81.0ms | 75.4ms | 49.6ms | × |
| YOLOv9t | Detect | GPU | FP16 | ? | 45.9ms | 33.5ms | ? | 41.5ms |
| YOLOv9t | Detect | CPU | INT8 | × | 74.4ms | × | 46.8ms | × |
| YOLOv9t | Detect | GPU | INT8 | × | 384.5ms | × | ? | 47.5ms |
| YOLOv10n | Detect | CPU | FP32 | 33.7ms | 34.7ms | × | 28.6ms | × |
| YOLOv10n | Detect | GPU | FP32 | 8.3ms | 13.0m | × | ? | × |
| YOLOv10n | Detect | CPU | FP16 | × | 57.8ms | × | 28.6ms | × |
| YOLOv10n | Detect | GPU | FP16 | ? | 14.4ms | × | ? | × |
| YOLOv10n | Detect | CPU | INT8 | × | 49.8ms | × | 26.1ms | × |
| YOLOv10n | Detect | GPU | INT8 | × | 103.0ms | × | ? | × |
| YOLOv11n | Classify | CPU | FP32 | 4.1ms | 2.3ms | 4.6ms | 2.5ms | × |
| YOLOv11n | Classify | GPU | FP32 | 2.8ms | 1.7ms | × | ? | 3.7ms |
| YOLOv11n | Classify | CPU | FP16 | × | 6.1ms | 4.5ms | 2.5ms | × |
| YOLOv11n | Classify | GPU | FP16 | ? | 1.9ms | × | ? | 3.3ms |
| YOLOv11n | Classify | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Classify | GPU | INT8 | × | ? | × | ? | 3.6ms |
| YOLOv11n | Detect | CPU | FP32 | 62.2ms | 52.9ms | 66.2ms | 45.2ms | × |
| YOLOv11n | Detect | GPU | FP32 | 38.7ms | 41.2ms | × | ? | 36.6ms |
| YOLOv11n | Detect | CPU | FP16 | × | 82.5ms | 63.0ms | 45.1ms | × |
| YOLOv11n | Detect | GPU | FP16 | ? | 58.2ms | × | ? | 38.2ms |
| YOLOv11n | Detect | CPU | INT8 | × | ? | × | 50.0ms | × |
| YOLOv11n | Detect | GPU | INT8 | × | ? | × | ? | 39.1ms |
| YOLOv11n | Segment | CPU | FP32 | 183.5ms | 152.7ms | 144.1ms | 91.9ms | × |
| YOLOv11n | Segment | GPU | FP32 | 98.2ms | 116.2ms | × | ? | 114.9ms |
| YOLOv11n | Segment | CPU | FP16 | × | 185.4ms | 155.2ms | 92.3ms | × |
| YOLOv11n | Segment | GPU | FP16 | ?ms | 130.4ms | × | ? | 120.2ms |
| YOLOv11n | Segment | CPU | INT8 | × | ? | × | ? | × |
| YOLOv11n | Segment | GPU | INT8 | × | ? | × | ? | 39.0ms |

You can get a docker image with:
```bash
docker pull taify/yolo_inference:latest
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