#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include "stereo_v3.hpp"
using namespace cv;
using namespace std;
vector<Point2f> readPts(int index)
{
   char filename[100];
   sprintf(filename, "../Data/Series/Set10/path%d.xml", index);
   ifstream file(filename,ios::in);
   vector<Point2f> vec;
   int n = 0;
   while (n++ < 20) 
   {
      Point2f p;
      file >> p.x;
      file >> p.y;
      vec.push_back(p);
   }
   return vec;
}
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);

   int left = atoi(args.left_image_name);
   int right = atoi(args.right_image_name);
   char fname[100];
   sprintf(fname, "../Data/Series/Set10/%d.JPG", left);
   Mat img1 = imread(fname, IMREAD_COLOR);
   sprintf(fname, "../Data/Series/Set10/%d.JPG", right);
   Mat img2 = imread(fname, IMREAD_COLOR);

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

   vector<Point2f> imgpts1 = readPts(left);
   vector<Point2f> imgpts2 = readPts(right);
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
      undistortPoints(imgpts1, imgpts1, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      undistortPoints(imgpts2, imgpts2, camera_matrix, dist_coefficients, noArray(), camera_matrix);
   } 

   /* Why not use these? */
   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat R, t;
   Mat mask; // inlier mask
   Mat E = findEssentialMat(imgpts1, imgpts2, focal, principalPoint, RANSAC, 0.1, 1000, mask);
   Mat F = camera_matrix.inv().t() * E * camera_matrix.inv();
   int inliers = recoverPose(E, imgpts1, imgpts2, R, t, focal, principalPoint, mask);

   cout << "Matches used for pose recovery: " << inliers << endl;

   /* Mat R1, R2, ProjMat1, ProjMat2, Q; */
   /* stereoRectify(camera_matrix, dist_coefficients, camera_matrix, dist_coefficients, img1.size(), R, t, R1, R2, ProjMat1, ProjMat2, Q, 0); */
   /* Mat new_camera_matrix, map11, map12, map21, map22, img1_remapped, img2_remapped; */
   /* initUndistortRectifyMap(camera_matrix, dist_coefficients, R1, ProjMat1, img1.size(), CV_16SC2, map11, map12); */
   /* initUndistortRectifyMap(camera_matrix, dist_coefficients, R2, ProjMat2, img2.size(), CV_16SC2, map21, map22); */
   /* remap(img1, img1_remapped, map11, map12, INTER_LINEAR, BORDER_CONSTANT); */
   /* remap(img2, img2_remapped, map21, map22, INTER_LINEAR, BORDER_CONSTANT); */
   /* Mat rectifiedImgs; */
   /* hconcat((Mat[]){img1_remapped, img2_remapped }, 2, rectifiedImgs); */
   /* imshow("foo", rectifiedImgs); */
   /* waitKey(0); */

   Mat mtxR, mtxQ;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
   cout << "Translation: " << t.t() << endl;
   cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

   if (args.epilines)
   {
      drawEpilines(Mat(imgpts1), 1, F, img2);
      drawEpilines(Mat(imgpts2), 2, F, img1);
   }

   if (args.draw_matches) 
   {
      Mat img_matches;
      drawMatches(img1, KeyPoints_1, img2, KeyPoints_2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), mask);
      namedWindow("Matches", CV_WINDOW_NORMAL);
      imshow("Matches", img_matches);
      waitKey(0);
   }

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;

   triangulatePoints(P1, P2, imgpts1, imgpts2, pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols


   double mDist = 0;
   int n = 0;
   int pos = 0, neg = 0;

   /* Write ply file header */
   ofstream ply_file("points.ply", ios_base::trunc);
   ply_file << 
      "ply\n"
      "format ascii 1.0\n"
      "element vertex " << dehomogenized.rows << "\n"
      "property float x\n"
      "property float y\n"
      "property float z\n"
      "property uchar red\n"
      "property uchar green\n"
      "property uchar blue\n"
      "end_header" << endl;

   Mat_<double> row;
   for (int i = 0; i < dehomogenized.rows; i++) 
   {
      row = dehomogenized.row(i);
      double d = row(2);
      if (d > 0) 
      {
         pos++;
         mDist += norm(row);
         n++;
         Vec3b rgb = img1.at<Vec3b>(imgpts1[i].x, imgpts1[i].y);
         ply_file << row(0) << " " << row(1) << " " << row(2) << " " << (int)rgb[2] << " " << (int)rgb[1] << " " << (int)rgb[0] << "\n";
      } else
      {
         neg++;
         ply_file << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << "\n"; 
      }
      Mat r = pnts4D.row(i).t();
      r.convertTo(r, P2.type());
      Mat projPt = P2 * r;
      projPt = projPt / projPt.at<double>(2);
      /* cout << projPt.t() << endl; */
   }

   mDist /= n;
   cout << "Mean distance of " << n << " points to camera: " << mDist << " (dehomogenized)" << endl;
   cout << "pos=" << pos << ", neg=" << neg << endl;
   return 0;
}
