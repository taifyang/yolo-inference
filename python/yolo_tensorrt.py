import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda  
from yolo import *
from utils import *


class YOLO_TensorRT(YOLO):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__()
        assert os.path.exists(model_path), "model not exists!"
        assert device_type == Device_Type.GPU, "only support GPU!"
        self.algo_type = algo_type
        self.model_type = model_type
        logger = trt.Logger(trt.Logger.ERROR)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.stream = cuda.Stream()
            
    @abstractclassmethod       
    def pre_process(self) -> None:
        pass
    
    @abstractclassmethod    
    def process(self) -> None:
        pass
    
    @abstractclassmethod         
    def post_process(self) -> None:
        pass
   
        
class YOLO_TensorRT_Classification(YOLO_TensorRT):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__(algo_type, device_type, model_type, model_path)
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        
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
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()  
            self.output = self.output_host.reshape(context.get_binding_shape(1)) 
            
    def post_process(self) -> None:
        output = np.squeeze(self.output).astype(dtype=np.float32)
        if self.algo_type == Algo_Type.YOLOv5:
            print("class:", np.argmax(output), " scores:", np.exp(np.max(output))/np.sum(np.exp(output)))
        if self.algo_type == Algo_Type.YOLOv8:
            print("class:", np.argmax(output), " scores:", np.max(output))
    
    
class YOLO_TensorRT_Detection(YOLO_TensorRT):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__(algo_type, device_type, model_type, model_path)
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  
        input = input / 255.0
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()  
            self.output = self.output_host.reshape(context.get_binding_shape(1)) 
        
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
        
             
class YOLO_TensorRT_Segmentation(YOLO_TensorRT):
    def __init__(self, algo_type:Algo_Type, device_type:Device_Type, model_type:Model_Type, model_path:str) -> None:
        super().__init__(algo_type, device_type, model_type, model_path)
        context = self.engine.create_execution_context()
        self.input_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        self.output0_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        self.output1_host = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output0_device = cuda.mem_alloc(self.output0_host.nbytes)
        self.output1_device = cuda.mem_alloc(self.output1_host.nbytes)
        
    def pre_process(self) -> None:
        input = letterbox(self.image, self.input_shape)
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  
        input = input / 255.0
        input = np.expand_dims(input, axis=0) 
        np.copyto(self.input_host, input.ravel())
        
    def process(self) -> None:
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            context.execute_async_v2(bindings=[int(self.input_device), int(self.output0_device), int(self.output1_device)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output0_host, self.output0_device, self.stream)
            cuda.memcpy_dtoh_async(self.output1_host, self.output1_device, self.stream)
            self.stream.synchronize()  
            self.output0 = self.output0_host.reshape(context.get_binding_shape(1)) 
            self.output1 = self.output1_host.reshape(context.get_binding_shape(2)) 
            
    def post_process(self) -> None:
        output1 = np.squeeze(self.output1).astype(dtype=np.float32)
        boxes = []
        scores = []
        class_ids = []
        preds = []
        if self.algo_type == Algo_Type.YOLOv5:
            output1 = output1[output1[..., 4] > self.confidence_threshold]
            classes_scores = output1[..., 5:85]     
            for i in range(output1.shape[0]):
                class_id = np.argmax(classes_scores[i])
                obj_score = output1[i][4]
                cls_score = classes_scores[i][class_id]
                output1[i][4] = obj_score * cls_score
                output1[i][5] = class_id
                if output1[i][4] > self.score_threshold:
                    boxes.append(output1[i][:6])
                    scores.append(output1[i][4])
                    class_ids.append(output1[i][5])   
                    output1[i][5:] *= obj_score
                    preds.append(output1[i])
        if self.algo_type == Algo_Type.YOLOv8: 
            for i in range(output1.shape[0]):
                classes_scores = output1[..., 4:84]     
                class_id = np.argmax(classes_scores[i])
                output1[i][4] = classes_scores[i][class_id]
                output1[i][5] = class_id
                if output1[i][4] > self.score_threshold:
                    boxes.append(output1[i, :6])
                    scores.append(output1[i][4])
                    class_ids.append(output1[i][5])   
                    preds.append(output1[i])            
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, self.score_threshold, self.nms_threshold) 
        boxes = boxes[indices]
        
        masks_in = np.array(preds)[indices][..., -32:]
        proto= np.squeeze(self.output0).astype(dtype=np.float32)
        c, mh, mw = proto.shape 
        masks = (1/ (1 + np.exp(-masks_in @ proto.reshape(c, -1)))).reshape(-1, mh, mw)
        
        downsampled_bboxes = boxes.copy()
        downsampled_bboxes[:, 0] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 2] *= mw / self.input_shape[0]
        downsampled_bboxes[:, 3] *= mh / self.input_shape[1]
        downsampled_bboxes[:, 1] *= mh / self.input_shape[1]
    
        masks = crop_mask(masks, downsampled_bboxes)
        self.result = draw(self.image, boxes, masks)