#include <opencv2/opencv.hpp>
#include <iostream>
#include <tclap/CmdLine.h>
#include <fstream>
#include <prettyprint/prettyprint.hpp>
#include "ImageSeries.hpp"
#include "serialization.hpp"
#include "CalibrationFileReader.h"

using namespace cv;
using namespace std;

void runEstimate(ImageSeries& series);
Vec3d rotationMatToEuler(Mat& R);

vector<string> getLinesFromFile(const string& fname, bool (*yesno)(string) = [](string s) { return true; })
{
   ifstream file(fname);
   vector<string> vec;
   string current_line;
   while (getline(file, current_line)) 
      if (yesno(current_line)) vec.push_back(current_line);
   return vec;
}

int main(int argc, const char *argv[])
{
   using TCLAP::CmdLine;
   using TCLAP::ValueArg;
   CmdLine cmd("Useful message", ' ', "0.1");
   ValueArg<string> images_arg("i", 
         "image-names",
         "Filenames of all images. Should be given as a path to a\
         file with newline-separated filenames. The first frame must\
         come first, the second frame second, the reference frame third.",
         true,
         "n/a",
         "File listing all image filenames");
   cmd.add(images_arg);

   ValueArg<string> correspondences_arg("c", 
         "correspondences",
         "Filenames of all files containing the correspondences. Should be given as a path to a\
         file with newline-separated filenames.\
         The line i will contain the filename for matches between first frame\
         and image i from the image file list, excluding the first frame itself.",
         true,
         "n/a",
         "File listing all correspondence filenames");
   cmd.add(correspondences_arg);

   ValueArg<string> calibration_arg("d",
         "calibration-data",
         "Filename of an OpenCV XML storage file with 'Camera_Matrix' and 'Distortion_Coefficients' nodes.",
         false,
         "n/a",
         "Calibration data file"
         );
   cmd.add(calibration_arg);

   cmd.parse(argc, argv);

   CalibrationFileReader reader(calibration_arg.getValue());
   vector<string> image_filenames = getLinesFromFile(images_arg.getValue(), [](string s){ return (bool)s.length(); });
   vector<string> correspondence_filenames = getLinesFromFile(correspondences_arg.getValue());
   if (image_filenames.size() != correspondence_filenames.size() + 1) 
   {
      cerr << "There must be one less correspondence filename than there are image file names." << endl;
      return -2;
   }
   if (image_filenames.size() <= 3)
   {
      cerr << "Error. Must be at least 4 files." << endl;
      return -1;
   }

   try
   {
      ImageSeries series(
            imread(image_filenames[0]), 
            imread(image_filenames[1]),
            imread(image_filenames[2]),
            reader.getCameraMatrix(),
            reader.getDistortionCoeffs()
            );
      std::cout << "Addding image " << image_filenames[0] << std::endl;
      std::cout << "Addding image " << image_filenames[1] << std::endl;
      std::cout << "Addding image " << image_filenames[2] << std::endl;
      for (auto iter = image_filenames.begin() + 3; iter != image_filenames.end(); iter++) 
      {
         std::cout << "Adding image " << *iter << std::endl;
         series.add_image(imread(*iter));
      }
      for (unsigned int i = 0; i < correspondence_filenames.size(); ++i) 
      {
          CorrVec&& corr = deserialize_vector<Point2i,Point2i>(correspondence_filenames[i]);
          std::cout << "Processing correspondences " << correspondence_filenames[i] << std::endl;
          switch (i)
          {
             case 0:
                series.add_correspondences(ImageSeries::SECOND_FRAME, corr); break;
             case 1:
                series.add_correspondences(ImageSeries::REF_FRAME, corr); break;
             default:
                series.add_correspondences(i-2, corr); break;
                break;
          }
      }
      std::cout << "Starting estimation..." << std::endl;
      runEstimate(series);
   } catch (std::exception& e)
   {
      cerr << "Caught exception: " << e.what() << endl;
      return -2;
   }

   return 0;
}

void runEstimate(ImageSeries& series)
{
   CorrVec& corr_first_second = series.correspondences_for_frame(ImageSeries::SECOND_FRAME);
   CorrVec& corr_first_ref = series.correspondences_for_frame(ImageSeries::REF_FRAME);
   const unsigned int n = corr_first_second.size();
   vector<KeyPoint> kpts_first(n), kpts_second(n), kpts_ref(n);
   vector<Point2f> pts_first(n), pts_second(n), pts_ref(n);
   convertToKeypoints(corr_first_second, kpts_first, kpts_second);
   convertToKeypoints(corr_first_ref, kpts_first, kpts_ref);
   for (unsigned int i = 0; i < n; ++i) 
   {
      pts_first[i] = kpts_first[i].pt;
      pts_second[i] = kpts_second[i].pt;
      pts_ref[i] = kpts_ref[i].pt;
   }
   Mat camera_matrix = series.camera_matrix();
   Mat dist_coeffs = series.dist_coeffs();

   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat mask;
   Mat E = findEssentialMat(pts_first, pts_second, focal, principalPoint, RANSAC, 0.999, 1, mask);

   Mat R, t;
   int inliers = recoverPose(E, pts_first, pts_second, R, t, focal, principalPoint, mask);
   Vec3d angles = rotationMatToEuler(R);
   /* std::cout << "Rotation: " << angles << std::endl; */
   /* std::cout << "Translation: " << t << std::endl; */
   
   vector<Point2f> pts_first_masked, pts_second_masked, pts_ref_masked;
   for (int i = 0; i < pts_first.size(); i++) 
   {
      if (mask.at<uchar>(i,0) == 1) 
      {
         pts_first_masked.push_back(pts_first[i]);
         pts_second_masked.push_back(pts_second[i]);
         pts_ref_masked.push_back(pts_ref[i]);
      }
   }

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;
   triangulatePoints(P1, P2, pts_first_masked, pts_second_masked, pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols

   /*** SOLVEPNP ***/
   Mat rvec, t_first_ref;
   solvePnP(dehomogenized, pts_ref_masked, camera_matrix, noArray(), rvec, t_first_ref);
   Mat R_first_ref;
   Rodrigues(rvec,R_first_ref);
   std::cout << "R_first_ref: " << rotationMatToEuler(R_first_ref) << std::endl;
   std::cout << "t_first_ref: " <<  t_first_ref << std::endl;

   for (unsigned int i = 0; i < 3; ++i) 
   {
      CorrVec corr_first_current = series.correspondences_for_frame(i);
      vector<Point2f> pts_current_masked;
      vector<KeyPoint> kpts_current(n);
      convertToKeypoints(corr_first_current, kpts_first, kpts_current);
      for (unsigned int i = 0; i < n; ++i) 
         if (mask.at<uchar>(i,0) == 1) 
            pts_current_masked.push_back(kpts_current[i].pt);
      Mat R_first_current, t_first_current;
      solvePnP(dehomogenized, pts_current_masked, camera_matrix, noArray(), rvec, t_first_current);
      Rodrigues(rvec,R_first_current);
      std::cout << "R_first_current: " << rotationMatToEuler(R_first_current) << std::endl;
      std::cout << "t_first_current: " <<  t_first_current << std::endl;
      series.showMatches(0, i + 3, series.correspondences_for_frame(i));
   }

}

Vec3d rotationMatToEuler(Mat& R)
{
   Mat mtxR, mtxQ;
   Mat Qx, Qy, Qz;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ, Qx, Qy, Qz);
   return angles;
}
