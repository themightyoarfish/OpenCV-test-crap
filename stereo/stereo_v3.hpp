/**
 * @file OCV3 compliant version of stereo_v2.cpp
 * @author Rasmus Diederichsen
 */
#ifndef STEREO_V3_H

#define STEREO_V3_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

typedef enum { DETECTOR_SURF, DETECTOR_KAZE } detector_type;

typedef struct 
{
   double minHessian = 800; // changing this dramatically affects the result, set lower if you want to see nothing 
   int nOctaves = 4;
   int nOctaveLayersSurf = 2;
   int nOctaveLayersAkaze = 4;
   bool extended = true;
   bool upright = false;
   int descriptor_size = 0;
   int descriptor_channels = 3;
   float threshold = .001;

} DetectorData;

typedef struct 
{
   char* left_image_name = "n/a";
   char* right_image_name = "n/a";
   char* calib_file_name = "n/a";
   bool use_ratio_test = false;
   float ratio = 0.8;
   int resize_factor = 1;
   bool undistort = true;
   bool epilines = false;
   detector_type detector = DETECTOR_KAZE;
   DetectorData detector_data;
   bool check_args()
   {
      if (0 == strcmp(left_image_name,"n/a") || 0 == strcmp(left_image_name, "n/a") || 0 == strcmp(calib_file_name, "n/a")) 
         return false;
      else return true;
   }

} CommandArgs;

void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas);
double computeReprojectionError(vector<Point2f>& img1pts, vector<Point2f>& img2pts, Mat& inlier_mask, const Mat& F);
ostream& operator<<(ostream& os, const DetectorData& d);
ostream& operator<<(ostream& os, const CommandArgs& d);
CommandArgs parse_args(int& argc, char* const* argv);
void computePoseDifference(Mat img1, Mat img2, CommandArgs args, Mat k, Mat& dist_coefficients, double& worldScale, Mat& R, Mat& t, Mat& img_matches);

#endif /* end of include guard: STEREO_V3_H */
