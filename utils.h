#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <sstream>


static cv::Vec3d rotationMatToEuler(const cv::Mat& R)
{
   using namespace cv;
   Mat mtxR, mtxQ;
   Mat Qx, Qy, Qz;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ, Qx, Qy, Qz);
   return angles;
}

namespace relative_pose
{

   struct PoseData {
      cv::Mat R;
      cv::Mat t;
      PoseData() : R(cv::Mat()), t(cv::Mat()) {}
      PoseData(cv::Mat& R, cv::Mat& t) : R(R), t(t) {}
      std::string to_string()
      {
         cv::Vec3d angles = rotationMatToEuler(R);
         std::stringstream sstream;
         sstream << angles << "\n";
         sstream << t.t();
         return sstream.str();
      }
   };


   typedef enum {DETECTOR_NONE, DETECTOR_SURF, DETECTOR_KAZE, DETECTOR_SIFT } detector_type;

}

#endif /* end of include guard: UTILS_H */
