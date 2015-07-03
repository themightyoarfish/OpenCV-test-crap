#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

#ifdef TIME
#include <chrono>
#endif

#include "stereo_v3.hpp"

int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   if (!args.check_args()) 
   {
      cout << "Usage: " << argv[0] << " --left IMG --right IMG2 --calib CALIB_FILE "
         "[--resize n] [--detector (KAZE|SURF) [--hessianT n] [--octaves n] [--octave-layers n] "
         "[--no-extend] [--upright] [--descriptor-size n] [--descriptor-channels {1,2,3}] [--threshold n]] "
         "[--epilines] [--no-undistort]" << endl;
      return -1;
   }
   Mat img1 = imread(args.left_image_name, IMREAD_COLOR);
   Mat img2 = imread(args.right_image_name, IMREAD_COLOR);

   if(!img1.data || !img2.data) 
   {
      cout << "At least one of the images has no data." << endl;
      return 1;
   }

   FileStorage fs(args.calib_file_name, FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      if (args.resize_factor > 1) 
      {
         resize(img1, img1, Size(img1.cols / args.resize_factor, 
                  img1.rows / args.resize_factor)); // make smaller for performance and displayablity
         resize(img2, img2, Size(img2.cols / args.resize_factor,
                  img2.rows / args.resize_factor));
         // scale matrix down according to changed resolution
         camera_matrix = camera_matrix / args.resize_factor;
         camera_matrix.at<double>(2,2) = 1;
      }

      Mat K1, K2;
      K1 = K2 = camera_matrix;
      if (img1.rows > img1.cols) // it is assumed the camera has been calibrated in landscape mode, so undistortion must also be performed in landscape orientation, or the camera matrix must be modified (fx,fy and cx,cy need to be exchanged)
      {
         swap(K1.at<double>(0,0), K1.at<double>(1,1));
         swap(K1.at<double>(0,2), K1.at<double>(1,2));
      }
      if (img2.rows > img2.cols)
      {
         swap(K2.at<double>(0,0), K2.at<double>(1,1));
         swap(K2.at<double>(0,2), K2.at<double>(1,2));
      }



      // Feature detection + extraction
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      Mat descriptors_1, descriptors_2;

      Ptr<Feature2D> feat_detector;
      if (args.detector == DETECTOR_KAZE) 
      {
         feat_detector = AKAZE::create(args.detector_data.upright ? AKAZE::DESCRIPTOR_MLDB_UPRIGHT : AKAZE::DESCRIPTOR_MLDB, 
               args.detector_data.descriptor_size,
               args.detector_data.descriptor_channels,
               args.detector_data.threshold,
               args.detector_data.nOctaves,
               args.detector_data.nOctaveLayersAkaze);

      } else 
      {
         feat_detector = xfeatures2d::SURF::create(args.detector_data.minHessian, 
               args.detector_data.nOctaves, args.detector_data.nOctaveLayersAkaze, args.detector_data.extended, args.detector_data.upright);
      }

#ifdef TIME
      chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
#endif

      feat_detector->detectAndCompute(img1, noArray(), KeyPoints_1, descriptors_1);
      feat_detector->detectAndCompute(img2, noArray(), KeyPoints_2, descriptors_2);

#ifdef TIME
      chrono::high_resolution_clock::time_point done_detection = chrono::high_resolution_clock::now();
      chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(done_detection - start);
      cout << "Detection took " << time_span.count() << " seconds." << endl;
#endif

      // Find correspondences
      BFMatcher matcher(args.detector == DETECTOR_KAZE ? NORM_HAMMING : NORM_L2, false);
      vector<DMatch> matches;

      vector<vector<DMatch>> match_candidates;
      const float ratio = .8; // Lowe
      matcher.knnMatch(descriptors_1, descriptors_2, match_candidates, 2);
      for (int i = 0; i < match_candidates.size(); i++)
      {
         if (match_candidates[i][0].distance < ratio * match_candidates[i][1].distance)
         {
            matches.push_back(match_candidates[i][0]);
         }
      }

#ifdef TIME
      chrono::high_resolution_clock::time_point done_matching = chrono::high_resolution_clock::now();
      time_span = chrono::duration_cast<chrono::duration<double>>(done_matching - done_detection);
      cout << "Matching took " << time_span.count() << "seconds." << endl;
      time_span = chrono::duration_cast<chrono::duration<double>>(done_matching - start);
      cout << "Complete procedure took " << time_span.count() << " seconds for " << KeyPoints_1.size() + KeyPoints_2.size() << " total keypoints" << " (" << matches.size() << " used)" << endl;
#endif

      cout << "Total number of matches " << matches.size() << endl;
      cout << "img1.size (cols,rows)=" << "(" << img1.cols << "," << img1.rows << ")" << endl;
      cout << "img2.size (cols,rows)=" << "(" << img2.cols << "," << img2.rows << ")" << endl;
      cout << "keypoints1 size=" << KeyPoints_1.size() << endl;
      cout << "keypoints2 size=" << KeyPoints_2.size() << endl;

      // Convert correspondences to vectors
      vector<Point2f>imgpts1,imgpts2;

      for(unsigned int i = 0; i < matches.size(); i++) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      if (args.undistort) 
      {
         undistortPoints(imgpts1, imgpts1, K1, dist_coefficients, noArray(), K1);
         undistortPoints(imgpts2, imgpts2, K2, dist_coefficients, noArray(), K2);
      } 
      Mat E = findEssentialMat(imgpts1, imgpts2, 1, Point2d(0,0), RANSAC, 0.999, 8, mask);
      correctMatches(E, imgpts1, imgpts2, imgpts1, imgpts2);

      Mat R, t; // rotation and translation
      cout << "Pose recovery inliers: " << recoverPose(E, imgpts1, imgpts2, R, t, 1.0, Point2d(0,0), mask) << endl;

      /* double theta_x, theta_y, theta_z; */
      /* theta_x = atan2(R.at<double>(2,1),  R.at<double>(2,2)); */
      /* theta_y = atan2(-R.at<double>(2,0), sqrt(pow(R.at<double>(2,1), 2) + pow(R.at<double>(2,2),2))); */
      /* theta_z = atan2(R.at<double>(1,0),  R.at<double>(0,0)); */
      /* cout << "\tx rotation: " << theta_x * 180 / M_PI << endl; */
      /* cout << "\ty rotation: " << theta_y * 180 / M_PI << endl; */
      /* cout << "\tz rotation: " << theta_z * 180 / M_PI << endl; */


      Mat mtxR, mtxQ;
      Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
      if(fabsf(determinant(R))-1.0 > 1e-07) 
         cout << "det(R) = " << determinant(R) << ". Not a valid rotation matrix, estimates will be useless." << endl;
      cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

      cout << "Translation [x y z]: " << t.t() << endl;

      double err = computeReprojectionError(imgpts1, imgpts2, mask, E);
      cout << "average reprojection err = " <<  err << endl;
      if (args.epilines)
      {
         drawEpilines(Mat(imgpts1), 1, E, img2);
         drawEpilines(Mat(imgpts2), 2, E, img1);
      }

      Mat img_matches; // side-by-side comparison
      drawMatches(img1, KeyPoints_1, img2, KeyPoints_2, // draw only inliers given by mask
            matches, img_matches, Scalar::all(-1), Scalar::all(-1), mask);
      // display
      namedWindow("Matches", CV_WINDOW_NORMAL);
      imshow("Matches", img_matches);
      waitKey(0);

      return 0;
   } else
   {
      cout << "Could not read file " << args.calib_file_name << endl;
      return -1;
   }
}
