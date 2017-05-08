#pragma once
#include "CL\cl.hpp"
#include "opencv2\opencv.hpp"

#include <fstream>
#include <iostream>

#define C_INPUT (1)
#define C_OUTPUT (1<<1)

class OpenCLHelper
{
	cl_int err;
	cl_int deviceType;
	cl::Platform cPlatform;
	cl::Device cDevice;
	cl::Program cProgram;
	cl::Context cContext;
	cl::Kernel cKernel;
	cl::CommandQueue cQueue;
public:
	OpenCLHelper();

	bool createProgram(const std::string &inFile, cl_int deviceType = CL_DEVICE_TYPE_CPU);
	
	cl::Image2D getImage2DMemoryObject(cv::Mat &inImage, bool &returnedErrorStatus, cl_int mode = C_INPUT);

	bool initializeKernel(const char *functionName);
	
	cl::Kernel getCurrentKernel();

	bool initializeQueue();

	bool enqueueWriteImage2D(const cv::Mat &inImage, cl::Image2D &inImageBuf);

	bool enqueueReadImage2D(cv::Mat &inImage, cl::Image2D &inImageBuf);

	bool enqueueNDKernelImage2D(const cv::Mat &inImage);

	// Error handling
	const char* getErrorString(cl_int error);
	bool processError(cl_int error, std::string errorMessage);
	~OpenCLHelper();
};


