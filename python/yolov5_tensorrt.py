import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda  
from yolov5 import *
from utils import *


class YOLOv5_TensorRT(YOLOv5):
    def __init__(self, model_path:str, device_type:Device_Type, model_type:Model_Type) -> None:
        super().__init__()
        assert device_type == Device_Type.GPU, "only support GPU!"
        self.model_type = model_type
        logger = trt.Logger(trt.Logger.ERROR)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        context = self.engine.create_execution_context()
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            self.inputs_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
            self.outputs_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        elif self.model_type == Model_Type.FP16:
            self.inputs_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float16)
            self.outputs_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float16)
        self.inputs_device = cuda.mem_alloc(self.inputs_host.nbytes)
        self.outputs_device = cuda.mem_alloc(self.outputs_host.nbytes)
        self.stream = cuda.Stream()
            
    def pre_process(self) -> None:
        input = letterbox(self.image, input_shape)
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  
        elif self.model_type == Model_Type.FP16:
            input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float16) 
        input = input / 255.0
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.inputs_host, input.ravel())
        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.inputs_device, self.inputs_host, self.stream)
            context.execute_async_v2(bindings=[int(self.inputs_device), int(self.outputs_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs_host, self.outputs_device, self.stream)
            self.stream.synchronize()  
            self.outputs = self.outputs_host.reshape(context.get_binding_shape(1)) 
    
    def post_process(self) -> None:
        self.outputs = np.squeeze(self.outputs).astype(dtype=np.float32)       
        self.outputs = self.outputs[self.outputs[..., 4] > confidence_threshold]
        classes_scores = self.outputs[..., 5:]       
        boxes = []
        scores = []
        class_ids = []
        for i in range(len(classes_scores)):
            class_id = np.argmax(classes_scores[i])
            self.outputs[i][4] *= classes_scores[i][class_id]
            self.outputs[i][5] = class_id
            if self.outputs[i][4] > score_threshold:
                boxes.append(self.outputs[i][:6])
                scores.append(self.outputs[i][4])
                class_ids.append(self.outputs[i][5])               
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, score_threshold, nms_threshold) 
        boxes = boxes[indices]
        draw(self.result, boxes)
        
        