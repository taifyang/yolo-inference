import openvino as ov
from yolo import *
from utils import *


class YOLO_OpenVINO(YOLO):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), "model not exists!"
        core = ov.Core()
        model  = core.read_model(model_path)
        self.algo_type = algo_type
        self.compiled_model = core.compile_model(model, device_name="GPU" if device_type==Device_Type.GPU else "CPU")
            
    def pre_process(self) -> None:
        input = letterbox(self.image, input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        self.inputs = np.expand_dims(input, axis=0)
        
    def process(self) -> None:
        results = self.compiled_model({0: self.inputs})
        self.outputs = results[self.compiled_model.output(0)]
             
    def post_process(self) -> None:
        self.outputs = np.squeeze(self.outputs).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        if self.algo_type == Algo_Type.YOLOv5:
            self.outputs = self.outputs[self.outputs[..., 4] > confidence_threshold]
            classes_scores = self.outputs[..., 5:]     
            for i in range(self.outputs.shape[0]):
                class_id = np.argmax(classes_scores[i])
                self.outputs[i][4] *= classes_scores[i][class_id]
                self.outputs[i][5] = class_id
                if self.outputs[i][4] > score_threshold:
                    boxes.append(self.outputs[i][:6])
                    scores.append(self.outputs[i][4])
                    class_ids.append(self.outputs[i][5])   
        if self.algo_type == Algo_Type.YOLOv8: 
            for i in range(self.outputs.shape[0]):
                classes_scores = self.outputs[..., 4:]     
                class_id = np.argmax(classes_scores[i])
                self.outputs[i][4] = classes_scores[i][class_id]
                self.outputs[i][5] = class_id
                if self.outputs[i][4] > score_threshold:
                    boxes.append(self.outputs[i, :6])
                    scores.append(self.outputs[i][4])
                    class_ids.append(self.outputs[i][5])               
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, score_threshold, nms_threshold) 
        boxes = boxes[indices]
        draw(self.result, boxes)

