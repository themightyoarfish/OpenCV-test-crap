#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "stereo_v3.hpp"
using namespace cv;
using namespace std;
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   Mat img1 = imread("../Data/Outdoor Samples/AUTO1.JPG", IMREAD_COLOR);
   Mat img2 = imread("../Data/Outdoor Samples/AUTO2.JPG", IMREAD_COLOR);

   if(!img1.data || !img2.data) 
   {
      cout << "At least one of the images has no data." << endl;
      return 1;
   }

   Mat_<double> camera_matrix(3,3);
   camera_matrix << 
      2.9880631668863380e+03, 0.,                     1.6315000000000000e+03, 
      0.,                     2.9880631668863380e+03, 1.2235000000000000e+03, 
      0.,                     0.,                     1.;
   Mat_<double> dist_coefficients(5,1);
   dist_coefficients << 
      1.4422094911174704e-01, -5.4684174329780899e-01,
      -7.5857781243513097e-04, 1.1949279901859115e-03,
      7.9061044687285797e-01;


#include "AUTO_pts.h"

   const int NPOINTS = imgpts1.size();
   if (args.resize_factor > 1) 
   {
      resize(img1, img1, Size(img1.cols / args.resize_factor, 
               img1.rows / args.resize_factor)); // make smaller for performance and displayablity
      resize(img2, img2, Size(img2.cols / args.resize_factor,
               img2.rows / args.resize_factor));
      // scale matrix down according to changed resolution
      camera_matrix = camera_matrix / args.resize_factor;
      camera_matrix.at<double>(2,2) = 1;
      for (int i = 0; i < NPOINTS; i++) 
      {
         imgpts1[i] = Point2f(imgpts1[i].x / args.resize_factor, imgpts1[i].y / args.resize_factor);
         imgpts2[i] = Point2f(imgpts2[i].x / args.resize_factor, imgpts2[i].y / args.resize_factor);
      }
   }


   vector<DMatch> matches(NPOINTS);
   vector<KeyPoint> KeyPoints_1(NPOINTS), KeyPoints_2(NPOINTS);
   for (int i = 0; i < NPOINTS; i++) 
   {
      matches[i] = DMatch(i,i,0);
      KeyPoints_1[i] = KeyPoint(imgpts1[i], 10);
      KeyPoints_2[i] = KeyPoint(imgpts2[i], 10);
   }

   if (args.undistort) 
   {
      undistortPoints(imgpts1, imgpts1, camera_matrix, dist_coefficients);
      undistortPoints(imgpts2, imgpts2, camera_matrix, dist_coefficients);
   } 

   /* Why not use these? */
   /* double focal = camera_matrix.at<double>(0,0); */
   /* Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2)); */

   Mat R, t;
   Mat mask; // inlier mask
   Mat E = findEssentialMat(imgpts1, imgpts2, 1., Point2d(0,0), LMEDS, 0.999, 1.0, mask);
   int inliers = recoverPose(E, imgpts1, imgpts2, R, t, 1., Point2d(0,0));

   cout << "Matches used for pose recovery: " << inliers << endl;

   /* Mat R1, R2, ProjMat1, ProjMat2, Q; */
   /* stereoRectify(camera_matrix, dist_coefficients, camera_matrix, dist_coefficients, img1.size(), R, t, R1, R2, ProjMat1, ProjMat2, Q); */
   /* cout << "P1=" << ProjMat1 << endl; */
   /* cout << "P2=" << ProjMat2 << endl; */
   /* cout << "Q=" << Q << endl; */

   Mat mtxR, mtxQ;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
   cout << "Translation: " << t.t() << endl;
   cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

   if (args.epilines)
   {
      drawEpilines(Mat(imgpts1), 1, E, img2);
      drawEpilines(Mat(imgpts2), 2, E, img1);
   }

   Mat img_matches;
   drawMatches(img1, KeyPoints_1, img2, KeyPoints_2, // draw only inliers given by mask
         matches, img_matches, Scalar::all(-1), Scalar::all(-1));
   if (args.draw_matches) 
   {
      namedWindow("Matches", CV_WINDOW_NORMAL);
      imshow("Matches", img_matches);
      waitKey(0);
   }

   vector<Point2f> imgpts1_masked, imgpts2_masked;
   for (int i = 0; i < NPOINTS; i++) 
   {
      if (mask.at<uchar>(i,0) == 1) 
      {
         imgpts1_masked.push_back(imgpts1[i]);
         imgpts2_masked.push_back(imgpts2[i]);
      }
   }

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;

   triangulatePoints(P1, P2, imgpts1_masked, imgpts2_masked, pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols


   double mDist = 0;
   int n = 0;
   int pos = 0, neg = 0;

   Mat_<double> row;
   for (int i = 0; i < dehomogenized.rows; i++) 
   {
      row = dehomogenized.row(i);
      double d = row(2);
      if (d > 0) 
      {
         pos++;
         mDist += d;
         n++;
      } else
      {
         neg++;
      }
   }
   mDist /= n;
   cout << "Mean distance of " << n << " points to camera: " << mDist << " (dehomogenized)" << endl;
   cout << "pos=" << pos << ", neg=" << neg << endl;
   return 0;
}
