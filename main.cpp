#include <stdio.h>
#include <signal.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <string>
#include <cmath>

//OpenCV libs:
//#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
//#include "cuda.hpp"
//#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace std;
bool STOP;


void my_handler(int s){
  printf("Caught signal %d\n",s);
  STOP = true;
}


int main(int argc, const char** argv)
{
  //CTRL+C signal handler
  STOP = false;
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
	
  //register gst debug info - moderate
  putenv("GST_DEBUG=*:3");
  bool useCamera = true;//parser.get<bool>("camera");
  bool update_bg_model = true;

  //gst-streamer kommando
  char* gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

  cout <<"GST command:"<<endl<< gst<< endl<<endl; 
  //open videostream
  VideoCapture cap(gst);
	
  //check if videostream opened nicely
  if( !cap.isOpened() )
    {
      printf("can not open camera or video file\n%s", "");
      return -1;
    }
  cout << "Opened camera!"<< endl;

  Mat frame;
     cap >> frame; // get a new frame from camera
  
  GpuMat d_lines;
  //Mat edges;
  GpuMat edges_gpu;
  Mat edges;
  namedWindow("edges",1);
  Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);
  Ptr<cuda::HoughCirclesDetector> houghcircles = cuda::createHoughCirclesDetector(1, frame.rows/4, 100, 200,100,1000,3);
  Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(0, 30, 3);
  
  while(!STOP)
    {
      //clock_t start = clock(); //fps counter start
      int64 start = getTickCount();
      cap >> frame; // get a new frame from camera
      cv::cvtColor(frame, edges, COLOR_BGR2GRAY);
      cv::GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
      //cv::Canny(edges, edges, 0, 30, 3);//using the cpu
      GpuMat edges1(edges);
      canny->detect(edges1,edges_gpu);
      //edges_gpu.download(edges);


      //Hough Segments:
      /*
      GpuMat d_src(edges_gpu);
      hough->detect(edges_gpu, d_lines);
      vector<Vec4i> lines_gpu;
      if (!d_lines.empty())
	{
	  lines_gpu.resize(d_lines.cols);
	  Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
	  d_lines.download(h_lines);
	}
      
      for (size_t i = 0; i < lines_gpu.size(); ++i)
	{
	  Vec4i l = lines_gpu[i];
	  line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
      */
      

      
      //Hough Circles:
      //GpuMat d_src(edges_gpu);
      GpuMat d_circles;
      houghcircles->detect(edges_gpu, d_circles);//(d_src,d_circles);
      vector<Vec3f> circles;
      //cout << "size of d_circles:"<<d_circles.size() << endl;
      if(!d_circles.empty())
	{
	  circles.resize(d_circles.cols);
	  //Mat h_lines(1, d_circles.cols, CV_32SC4, &circles[0]);
	  d_circles.download(circles);
	}
      for( size_t i = 0; i < circles.size(); i++ )
	{

	  Vec3i c = circles[i];
	  circle( frame, Point(c[0], c[1]), c[2], Scalar(0,0,255), 3, LINE_AA);
	  circle( frame, Point(c[0], c[1]), 2, Scalar(0,255,0), 3, LINE_AA);
	  //Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	  //int radius = cvRound(circles[i][2]);
	  // draw the circle center
	  //circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
	  // draw the circle outline
	  //circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
      


      
      double dur = (getTickCount() - start)/getTickFrequency(); 
      double fps = 0.0;
      if(dur!=0.0)
	fps = 1/dur;
      //cout << "dur = "<<dur<<", fps = "<< fps << endl; 
      string x = "fps" + to_string(fps);
      //cout << "string=" << x << endl;
      putText(frame,//edges if you wanna display edges 
	      x, 
	      Point(edges.cols/10, edges.rows/10), 
	      FONT_HERSHEY_DUPLEX,
	      1.0,
	      CV_RGB(255,255,255),
	      2);

      //imshow("edges", edges);
      imshow("edges", frame);
      waitKey(1);
    }
  cap.release();
  destroyAllWindows();
  cout <<endl<< "Released Camera!" << endl;
  return 0;
}
