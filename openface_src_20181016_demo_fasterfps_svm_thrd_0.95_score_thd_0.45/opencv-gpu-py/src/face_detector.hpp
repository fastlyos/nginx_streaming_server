#ifndef FACEDETECTOR_HPP
#define FACEDETECTOR_HPP

#include "opencv_version.hpp"

#ifdef OPENCV_2
  #include <opencv2/core/core.hpp>           // Rect
  #include <opencv2/gpu/gpu.hpp>             // gpu::CascadeClassifier_GPU
  #include <opencv2/objdetect/objdetect.hpp> // CascadeClassifier
  #include "opencv2/opencv.hpp"
#else
  #include <opencv2/core.hpp>
  #include <opencv2/objdetect.hpp>           // CascadeClassifier
  #include <opencv2/cudaobjdetect.hpp>       // cuda::CascadeClassifier
  #include <opencv2/videoio.hpp>
#endif


class FaceDetector
{
 public:
  FaceDetector();
  void Init(const std::string& cascade_file, bool gpu);
  //bool Detect(const char* image_file, std::vector<cv::Rect>& face_rects);
  bool Detect(cv::Mat frame, std::vector<cv::Rect>& face_rects);
 private:
  bool gpu_;
  std::string cascade_file_;
  cv::VideoCapture vcap;
};


#endif
