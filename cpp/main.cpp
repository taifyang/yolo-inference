/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-11-15 10:26:20
 * @FilePath: \cpp\main.cpp
 * @Description: demo
 */
#include "yolo.h"


int main(int argc, char* argv[])
{
	if (argc != 7)
	{
		std::cerr << "argc input error" << std::endl;
		return -1;
	}
	std::cout << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5] << " " << argv[6] << std::endl;
	Backend_Type backend;
	Task_Type task;
	Algo_Type algo;
	Device_Type device;
	Model_Type model;
	std::string model_path = argv[6];

	try
	{
		backend = magic_enum::enum_cast<Backend_Type>(argv[1]).value();
		task = magic_enum::enum_cast<Task_Type>(argv[2]).value();
		algo = magic_enum::enum_cast<Algo_Type>(argv[3]).value();
		device = magic_enum::enum_cast<Device_Type>(argv[4]).value();
		model = magic_enum::enum_cast<Model_Type>(argv[5]).value();
	}
	catch (const std::bad_optional_access& e)
	{
        std::cerr << "argv input error: " << e.what() << std::endl;
		return -1;
    }

	std::unique_ptr<YOLO> yolo = CreateFactory::instance().create(backend, task);
	yolo->init(algo, device, model, model_path);
	yolo->infer("bus.jpg", false, false, argv);
	yolo->release();
	return 0;
}

