#include "yolo.h"


int main(int argc, char* argv[])
{
	std::unique_ptr<YOLO> yolo = BackendFactory::instance().create(Backend_Type(atoi(argv[1])));
	yolo->init(Algo_Type(atoi(argv[2])), Device_Type(atoi(argv[3])), Model_Type(atoi(argv[4])), argv[5]);
	yolo->infer("test.mp4", argv);
	yolo->release();

	return 0;
}