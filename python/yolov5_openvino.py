from openvino.inference_engine import IECore
from yolov5 import *
from utils import *


class YOLOv5_OpenVINO(YOLOv5):
    def __init__(self, model_path:str, device_type:Device_Type, model_type:Model_Type) -> None:
        super().__init__()
        ie = IECore()
        net = ie.read_network(model=model_path)
        self.exec_net = ie.load_network(network=net, device_name="GPU" if device_type==Device_Type.GPU else "CPU")
        self.input_layer = next(iter(net.input_info))
            
    def pre_process(self) -> None:
        input = letterbox(self.image, input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        self.inputs = np.expand_dims(input, axis=0)
        
    def process(self) -> None:
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs={self.input_layer: self.inputs})
        if infer_request_handle.wait(-1) == 0:
            output_layer = infer_request_handle._outputs_list[0]
            self.outputs = infer_request_handle.output_blobs[output_layer].buffer
             
    def post_process(self) -> None:
        self.outputs = np.squeeze(self.outputs)
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

