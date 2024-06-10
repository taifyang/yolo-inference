import onnxruntime
from yolo import *
from utils import *


class YOLO_ONNXRuntime(YOLO):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), "model not exists!"
        if device_type == Device_Type.CPU:
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        elif device_type == Device_Type.GPU:
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.algo_type = algo_type
        self.model_type = model_type
         
        self.input_name = []
        for node in self.onnx_session.get_inputs(): 
            self.input_name.append(node.name)
        self.output_name = []
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)
        self.input = {}
    
    @abstractclassmethod       
    def pre_process(self) -> None:
        pass
        
    def process(self) -> None:
        self.output = self.onnx_session.run(None, self.input)
    
    @abstractclassmethod         
    def post_process(self) -> None:
        pass


class YOLO_ONNXRuntime_Classification(YOLO_ONNXRuntime):           
    def pre_process(self) -> None:
        if self.algo_type == Algo_Type.YOLOv5:
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
            input = input - np.array([0.406, 0.456, 0.485])
            input = input / np.array([0.225, 0.224, 0.229])
        if self.algo_type == Algo_Type.YOLOv8:
            self.input_shape = (224, 224)
            if self.image.shape[1] > self.image.shape[0]:
                self.image = cv2.resize(self.image, (self.input_shape[0]*self.image.shape[1]//self.image.shape[0], self.input_shape[0]))
            else:
                self.image = cv2.resize(self.image, (self.input_shape[1], self.input_shape[1]*self.image.shape[0]//self.image.shape[1]))
            crop_size = min(self.image.shape[0], self.image.shape[1])
            left = (self.image.shape[1] - crop_size) // 2
            top = (self.image.shape[0] - crop_size) // 2
            crop_image = self.image[top:(top+crop_size), left:(left+crop_size), ...]
            input = cv2.resize(crop_image, self.input_shape)
            input = input / 255.0
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == Model_Type.FP16:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
            
    def post_process(self) -> None:
        output = np.squeeze(self.output).astype(dtype=np.float32)
        if self.algo_type == Algo_Type.YOLOv5:
            print("class:", np.argmax(output), " scores:", np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type == Algo_Type.YOLOv8:
            print("class:", np.argmax(output), " scores:", np.max(output))


class YOLO_ONNXRuntime_Detection(YOLO_ONNXRuntime):
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == Model_Type.FP16:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
            
    def post_process(self) -> None:
        output = np.squeeze(self.output[0]).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        if self.algo_type == Algo_Type.YOLOv5:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:85]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output[i][4]
                cls_score = classes_scores[i][class_id]
                output[i][4] = obj_score * cls_score
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i][:6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])   
                    output[i][5:] *= obj_score
        if self.algo_type == Algo_Type.YOLOv8: 
            for i in range(output.shape[0]):
                classes_scores = output[..., 4:]     
                class_id = np.argmax(classes_scores[i])
                output[i][4] = classes_scores[i][class_id]
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i, :6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])                  
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
        boxes = boxes[indices]
        self.result = draw(self.image, boxes)
        
        
class YOLO_ONNXRuntime_Segmentation(YOLO_ONNXRuntime):
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        if self.model_type == Model_Type.FP32 or self.model_type == Model_Type.INT8:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float32)
        elif self.model_type == Model_Type.FP16:
            input = np.expand_dims(input, axis=0).astype(dtype=np.float16)
        for name in self.input_name:
            self.input[name] = input
            
    def post_process(self) -> None:
        output = np.squeeze(self.output[0]).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type == Algo_Type.YOLOv5:
            output = output[output[..., 4] > self.confidence_threshold]
            classes_scores = output[..., 5:85]     
            for i in range(output.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output[i][4]
                cls_score = classes_scores[i][class_id]
                output[i][4] = obj_score * cls_score
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i][:6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])   
                    output[i][5:] *= obj_score
                    preds.append(output[i])
        if self.algo_type == Algo_Type.YOLOv8: 
            for i in range(output.shape[0]):
                classes_scores = output[..., 4:84]     
                class_id = np.argmax(classes_scores[i])
                output[i][4] = classes_scores[i][class_id]
                output[i][5] = class_id
                if output[i][4] > self.score_threshold:
                    boxes.append(output[i, :6])
                    scores.append(output[i][4])
                    class_ids.append(output[i][5])    
                    preds.append(output[i])           
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
        boxes = boxes[indices]
        
        masks_in = np.array(preds)[indices][..., -32:]
        proto= np.squeeze(self.output[1]).astype(dtype=np.float32)
        c, mh, mw = proto.shape 
        masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)
        
        downsampled_bboxes = boxes.copy()
        downsampled_bboxes[:, 0] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 2] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 3] *= mh / self.input_shape[1]
        downsampled_bboxes[:, 1] *= mh / self.input_shape[1]
    
        masks = crop_mask(masks, downsampled_bboxes)
        self.result = draw(self.image, boxes, masks)