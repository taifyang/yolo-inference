import cv2
from yolov5 import *
from utils import *


class YOLOv5_OpenCV(YOLOv5):
    def __init__(self, model_path:str, device_type:Device_Type, model_type:Model_Type) -> None:
        super().__init__()
        self.net = cv2.dnn.readNet(model_path)
        if device_type == Device_Type.GPU:
            self.net.setPreferableBackend(cv2.dnn_DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        assert model_type == Model_Type.FP32, "unsupported model type!"
            
    def pre_process(self) -> None:
        input = letterbox(self.image, input_shape)
        self.inputs = cv2.dnn.blobFromImage(input, 1/255., size=input_shape, swapRB=True, crop=False)
        self.net.setInput(self.inputs)
        
    def process(self) -> None:
        self.outputs = self.net.forward()
        
    def post_process(self) -> None:
        boxes = []
        scores = []
        class_ids = []
        for i in range(self.outputs.shape[1]):
            data = self.outputs[0][i]
            objness = data[4]
            if objness > confidence_threshold:
                score = data[5:] * objness
                _, _, _, max_score_index = cv2.minMaxLoc(score)
                max_id = max_score_index[1]
                if score[max_id] > score_threshold:
                    x, y, w, h = data[0].item(), data[1].item(), data[2].item(), data[3].item()
                    boxes.append(np.array([x-w/2, y-h/2, x+w/2, y+h/2]))
                    scores.append(score[max_id]*objness)
                    class_ids.append(max_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
        output = []
        for i in indices:
            output.append(np.array([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], class_ids[i]]))
        boxes = np.array(output)
        draw(self.result, boxes)

