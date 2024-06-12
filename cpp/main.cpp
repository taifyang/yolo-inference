#include "yolo.h"

int main(int argc, char* argv[])
{
	std::unique_ptr<YOLO> yolo = CreateFactory::instance().create(Backend_Type(atoi(argv[1])), Task_Type(atoi(argv[2])));
	yolo->init(Algo_Type(atoi(argv[3])), Device_Type(atoi(argv[4])), Model_Type(atoi(argv[5])), argv[6]);
	yolo->infer("bus.jpg", argv, true, false);
	yolo->release();
	return 0;

	//std::unique_ptr<YOLO> yolo = CreateFactory::instance().create(Backend_Type::TensorRT, Task_Type::Segment);
	//yolo->init(Algo_Type::YOLOv5, Device_Type::GPU, Model_Type::FP32, "yolov5n_seg_fp32.engine");
	//yolo->infer("bus.jpg", argv, true, true);
	//yolo->release();
	//return 0;
}

