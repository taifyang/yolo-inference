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
            
    @abstractclassmethod       
    def pre_process(self) -> None:
        pass
        
    def process(self) -> None:
        self.output = self.compiled_model({0: self.input})

             
    @abstractclassmethod         
    def post_process(self) -> None:
        pass


class YOLO_OpenVINO_Classification(YOLO_OpenVINO):
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
        self.input = np.expand_dims(input, axis=0)
            
    def post_process(self) -> None:
        output = self.output[self.compiled_model.output(0)]
        output = np.squeeze(output).astype(dtype=np.float32)
        if self.algo_type == Algo_Type.YOLOv5:
            print("class:", np.argmax(output), " scores:", np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type == Algo_Type.YOLOv8:
            print("class:", np.argmax(output), " scores:", np.max(output))
       
    
class YOLO_OpenVINO_Detection(YOLO_OpenVINO):
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        self.input = np.expand_dims(input, axis=0)
        
    def post_process(self) -> None:
        output = self.output[self.compiled_model.output(0)]
        output = np.squeeze(output).astype(dtype=np.float32)
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
        
        
class YOLO_OpenVINO_Segmentation(YOLO_OpenVINO):
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB和HWC2CHW
        input = input / 255.0
        self.input = np.expand_dims(input, axis=0)
            
    def post_process(self) -> None:
        output0 = self.output[self.compiled_model.output(0)]
        output0 = np.squeeze(output0).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type == Algo_Type.YOLOv5:
            output0 = output0[output0[..., 4] > self.confidence_threshold]
            classes_scores = output0[..., 5:85]     
            for i in range(output0.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output0[i][4]
                cls_score = classes_scores[i][class_id]
                output0[i][4] = obj_score * cls_score
                output0[i][5] = class_id
                if output0[i][4] > self.score_threshold:
                    boxes.append(output0[i][:6])
                    scores.append(output0[i][4])
                    class_ids.append(output0[i][5])   
                    output0[i][5:] *= obj_score
                    preds.append(output0[i])
        if self.algo_type == Algo_Type.YOLOv8: 
            for i in range(output0.shape[0]):
                classes_scores = output0[..., 4:84]     
                class_id = np.argmax(classes_scores[i])
                output0[i][4] = classes_scores[i][class_id]
                output0[i][5] = class_id
                if output0[i][4] > self.score_threshold:
                    boxes.append(output0[i, :6])
                    scores.append(output0[i][4])
                    class_ids.append(output0[i][5])      
                    preds.append(output0[i])         
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
        boxes = boxes[indices]
        
        masks_in = np.array(preds)[indices][..., -32:]
        output1 = self.output[self.compiled_model.output(1)]
        proto = np.squeeze(output1).astype(dtype=np.float32)
        c, mh, mw = proto.shape 
        masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)
        
        downsampled_bboxes = boxes.copy()
        downsampled_bboxes[:, 0] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 2] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 3] *= mh / self.input_shape[1]
        downsampled_bboxes[:, 1] *= mh / self.input_shape[1]
    
        masks = crop_mask(masks, downsampled_bboxes)
        self.result = draw(self.image, boxes, masks)