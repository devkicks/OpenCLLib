#include "OpenCLHelper.h"

OpenCLHelper::OpenCLHelper()
{
	// Initializing everythitng
	err = 0;
	deviceType = CL_DEVICE_TYPE_CPU;
	
	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);
	processError(err, "Getting list of available platforms");

	cPlatform = platforms.front();
	std::vector<cl::Device> devices;
	err = cPlatform.getDevices(deviceType, &devices);
	processError(err, "Getting devices for specified device type");

	cDevice = devices.front();
	cContext = cl::Context(cDevice);
}

bool OpenCLHelper::createProgram(const std::string &inFile, cl_int deviceType)
{
	bool returnErrorStatus = true;
	std::ifstream kernelFile(inFile);
	if (kernelFile.fail())
	{
		std::cout << "Unable to open cl file" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	
	cProgram = cl::Program(cContext, sources, &err);
	returnErrorStatus &= processError(err, "Initializing Program");

	err = cProgram.build("-cl-std=CL1.2");
	returnErrorStatus &= processError(err, "Compile CL kernel file");

	return returnErrorStatus;
}

//cl::Context OpenCLHelper::getContextFromProgram(const cl::Program &program)
//{
//	return program.getInfo<CL_PROGRAM_CONTEXT>();
//}
//
//cl::Device OpenCLHelper::getDeviceFromContext(const cl::Context &context)
//{
//	return context.getInfo<CL_CONTEXT_DEVICES>().front();
//}

cl::Image2D OpenCLHelper::getImage2DMemoryObject(cv::Mat &inImage, bool &returnedErrorStatus, cl_int mode)
{
	returnedErrorStatus = true;

	cl::Image2D clImage;
	cl_int err = 0;
	if (mode == C_INPUT)
	{
		clImage = cl::Image2D(cContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_UNORM_INT8), inImage.cols, inImage.rows, 0, inImage.data, &err);
		returnedErrorStatus &= processError(err, "Initializing memory for input image buffer");
	}
	else if(mode == C_OUTPUT)
	{
		clImage = cl::Image2D(cContext, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_UNORM_INT8), inImage.cols, inImage.rows, 0, nullptr, &err);
		returnedErrorStatus &= processError(err, "Initializing memory for output image buffer");
	}
	return clImage;
}

bool OpenCLHelper::initializeKernel(const char *functionName)
{
	cKernel = cl::Kernel(cProgram, functionName, &err);
	return processError(err, "Linking compiled kernel");
}
cl::Kernel OpenCLHelper::getCurrentKernel()
{
	return cKernel;
}

bool OpenCLHelper::initializeQueue()
{
	err = 0;
	cQueue = cl::CommandQueue(cContext, cDevice, 0, &err);
	return processError(err, "Initialize Command Queue");
}

bool OpenCLHelper::enqueueWriteImage2D(const cv::Mat &inImage, cl::Image2D &inImageBuf)
{
	err = 0;
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[0] = 0;
	origin[0] = 0;

	cl::size_t<3> desc;
	desc[0] = inImage.cols;
	desc[1] = inImage.rows;
	desc[2] = 1;

	err = cQueue.enqueueWriteImage(inImageBuf, CL_TRUE, origin, desc, 0, 0, inImage.data);
	return processError(err, "Enqueue Write Image 2D");
}

bool OpenCLHelper::enqueueReadImage2D(cv::Mat &inImage, cl::Image2D &inImageBuf)
{
	err = 0;
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[0] = 0;
	origin[0] = 0;

	cl::size_t<3> desc;
	desc[0] = inImage.cols;
	desc[1] = inImage.rows;
	desc[2] = 1;

	err = cQueue.enqueueReadImage(inImageBuf, CL_TRUE, origin, desc, 0, 0, inImage.data);
	return processError(err, "Enqueue Read Image 2D");
}

bool OpenCLHelper::enqueueNDKernelImage2D(const cv::Mat &inImage)
{
	err = 0;
	err = cQueue.enqueueNDRangeKernel(this->cKernel, cl::NullRange, cl::NDRange(inImage.cols, inImage.rows), cl::NullRange);
	return processError(err, "Enqueue ND Kernel for Image 2D");
	
}

//template <typename T>
//bool OpenCLHelper::setKernelArgs(cl_int argIndex, T argVal)
//{
//	//cKernel.setArg<T>(argIndex, argVal);
//}
/*
cl::Buffer OpenCLHelper::getBufferMemoryObject(cl_int sizeOfBuffer, cl_int mode)
{
	cl::Buffer clBuffer;
	cl_int err = 0;
	if (mode == C_INPUT)
	{
		clBuffer = cl::Buffer(cContext, CL_MEM_READ_ONLY, sizeOfBuffer, nullptr, &err);
		processError(err, "Initializing memory for input buffer");
	}
	else if (mode == C_OUTPUT)
	{
		clBuffer = cl::Buffer(cContext, CL_MEM_WRITE_ONLY, sizeOfBuffer, nullptr, &err);
		processError(err, "Initializing memory for output buffer");
	}
	return clBuffer;
}*/

const char* OpenCLHelper::getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

// Function for processing any error messages
bool OpenCLHelper::processError(cl_int error, std::string errorMessage)
{
	if (error != 0)
	{
		std::cout << "OpenCLError:" << errorMessage << std::endl;
		std::cout << getErrorString(error) << std::endl;
		//exit(EXIT_FAILURE);
	}
	else
	{
		std::cout << "OpenCLSuccess:" << errorMessage << std::endl;
	}

	return (error == 0);
}

OpenCLHelper::~OpenCLHelper()
{
}
