import onnxruntime
from yolov5 import *
from utils import *


class YOLOv5_ONNXRuntime(YOLOv5):
    def __init__(self, model_path:str, device_type:Device_Type, model_type:Model_Type) -> None:
        super().__init__()
        assert os.path.exists(model_path)
        if device_type == Device_Type.CPU:
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        elif device_type == Device_Type.GPU:
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        
        self.model_type = model_type
         
        self.input_name = []
        for node in self.onnx_session.get_inputs(): 
            self.input_name.append(node.name)
        self.output_name = []
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)
        self.inputs = {}
            
    def pre_process(self) -> None:
        input = letterbox(self.image, input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == Model_Type.FP16:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.inputs[name] = input
        
    def process(self) -> None:
        self.outputs = self.onnx_session.run(None, self.inputs)
                
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

