#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

typedef struct {
   cv::Mat R;
   cv::Mat t;
} PoseData;


static cv::Vec3d rotationMatToEuler(const cv::Mat& R)
{
   using namespace cv;
   Mat mtxR, mtxQ;
   Mat Qx, Qy, Qz;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ, Qx, Qy, Qz);
   return angles;
}

typedef enum {DETECTOR_NONE, DETECTOR_SURF, DETECTOR_KAZE, DETECTOR_SIFT } detector_type;

#endif /* end of include guard: UTILS_H */
