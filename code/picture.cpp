#include "PCN.h"

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
	CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG
#pragma comment( lib, cvLIB("world") )
#endif //_WIN32



int main()
{
    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel",
                 "model/PCN-Tracking.prototxt");
    /// detection
    detector.SetMinFaceSize(20);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetDetectionThresh(0.37, 0.43, 0.97);
    /// tracking
    detector.SetTrackingPeriod(30);
    detector.SetTrackingThresh(0.95);
    detector.SetVideoSmooth(false);


	char path[256] = { 0 };
	sprintf_s(path, "%s%03d%s", "D:\\Data\\300w\\300VW_Dataset_2015_12_14\\", 1, "\\vid.avi");

	cv::VideoCapture capture(path);
	cv::Mat img;
	while (true)
	{
		capture >> img;
		if (img.empty())
			break;
		int64 t1 = cv::getTickCount();
        std::vector<Window> faces = detector.Detect(img);
		int64 t2 = cv::getTickCount();
		int64 t = (t2 - t1)*1000.0f / cv::getTickFrequency();
		std::cout << "Time Cost: " <<
			int(t) << " ms" << std::endl;
        for (int j = 0; j < faces.size(); j++)
        {
            DrawFace(img, faces[j]);
        }
		cv::imshow("PCN", img);
		cv::waitKey(1);
	}
	return 0;
}