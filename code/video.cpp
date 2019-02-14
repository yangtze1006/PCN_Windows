#include "PCN.h"
#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>
#include <opencv2/core.hpp>
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
	bool bExitLoop = false;
	int nFileIdx = 0;
	int nSkippedNum = 0;
	while (!bExitLoop)
	{
		char path[256] = { 0 };
		sprintf_s(path, "%s%03d%s", "D:\\Data\\300w\\300VW_Dataset_2015_12_14\\", nFileIdx++, "\\vid.avi");
		cv::VideoCapture capture(path);
		if (!capture.isOpened()) {
			std::cout << "can not open file:" << path << std::endl;
			nSkippedNum++;
			continue;
		}

		PCN detector("model/PCN.caffemodel",
			"model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
			"model/PCN-Tracking.caffemodel",
			"model/PCN-Tracking.prototxt");
		/// detection
		detector.SetMinFaceSize(40);
		detector.SetImagePyramidScaleFactor(1.414f);
		detector.SetDetectionThresh(0.37f, 0.43f, 0.97f);
		/// tracking
		detector.SetTrackingPeriod(30);
		detector.SetTrackingThresh(0.95);
		detector.SetVideoSmooth(true);
		cv::Mat img;
		cv::Mat resizeMat;
		int minFps = 10000;
		int nFrameCnt = 0;
		int fpsTotal = 0;
		int meanFps = 0;
		while (1)
		{
			capture >> img;
			if (img.empty()) {
				break;
			}
			nFrameCnt++;
			//cv::resize(img, resizeMat, cv::Size(640, 480));

			int64 t1 = cv::getTickCount();
			std::vector<Window> faces = detector.DetectTrack(img);
			int64 t2 = cv::getTickCount();
			int64 t = (t2 - t1)*1000.0f / cv::getTickFrequency();
			int fps = 1000.0 / t;
			std::stringstream ss;
			std::stringstream ss2;
			fpsTotal += fps;
			if (nFrameCnt % 100 == 0)
			{
				meanFps = fpsTotal / nFrameCnt;
			}
			ss << meanFps;

			cv::putText(img, ss.str() + "FPS",
				cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));

			for (int i = 0; i < faces.size(); i++)
			{
				DrawFace(img, faces[i]);
			}
			cv::imshow("PCN", img);
			char k = cv::waitKey(30);
			if (k == 27)
				break;
			else if (k == 'q' ||  k == 'Q')
			{
				bExitLoop = true;
				break;
			}
			else if (k == 'p' || k == 'P')
			{
				nFileIdx = nFileIdx - 2 - nSkippedNum;
				nFileIdx = nFileIdx < 0?0 : nFileIdx;
				break;
			}
		}

		capture.release();
		cv::destroyAllWindows();
		nSkippedNum = 0;
	}

    return 0;
}
