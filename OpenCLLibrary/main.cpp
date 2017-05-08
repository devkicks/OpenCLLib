#include "OpenCLHelper.h"
#include <fstream>
#include "CL\cl.hpp"
#include "opencv2\opencv.hpp"


int main()
{
	// first read the image to process
	cv::Mat inImage = cv::imread("Images\\testImage3.png", CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << inImage.channels() << std::endl;
	if (inImage.empty())
	{
		std::cout << "Input Image is empty" << std::endl;
		std::cout << "Creating dummy image" << std::endl;

		inImage = cv::Mat::zeros(480, 640, CV_8UC1);
	}

	// create a container to store output image
	cv::Mat outImage = cv::Mat::zeros(inImage.rows, inImage.cols, CV_8UC1);

	OpenCLHelper myOpenCLObj;
	bool returnedStatus = true; // to get the status;
	myOpenCLObj.createProgram("kernelfile.cl");

	cl::Image2D inImageBuf = myOpenCLObj.getImage2DMemoryObject(inImage, returnedStatus, C_INPUT);
	cl::Image2D outImageBuf = myOpenCLObj.getImage2DMemoryObject(outImage, returnedStatus, C_OUTPUT);
	
	myOpenCLObj.initializeKernel("image_flip");
	cl::Kernel currentKernel = myOpenCLObj.getCurrentKernel();
	currentKernel.setArg(0, inImageBuf);
	currentKernel.setArg(1, outImageBuf);

	myOpenCLObj.initializeQueue();
	myOpenCLObj.enqueueWriteImage2D(inImage, inImageBuf);
	myOpenCLObj.enqueueNDKernelImage2D(inImage);
	myOpenCLObj.enqueueReadImage2D(outImage, outImageBuf);

	cv::imshow("outImage!", outImage);
	cv::waitKey(0);

	std::cin.get();
}
