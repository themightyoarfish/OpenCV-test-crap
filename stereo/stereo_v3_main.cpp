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
   Mat img_1 = imread(args.left_image_name, IMREAD_COLOR);
   Mat img_2 = imread(args.right_image_name, IMREAD_COLOR);

   FileStorage fs(args.calib_file_name, FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients, img_1_undist, img_2_undist;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      img_1_undist = img_1;
      img_2_undist = img_2;
      if (args.resize_factor > 1) 
      {
         resize(img_1_undist, img_1_undist, Size(img_1_undist.cols / args.resize_factor, 
                  img_1_undist.rows / args.resize_factor)); // make smaller for performance and displayablity
         resize(img_2_undist, img_2_undist, Size(img_2_undist.cols / args.resize_factor,
                  img_2_undist.rows / args.resize_factor));
         // scale matrix down according to changed resolution
         camera_matrix = camera_matrix / args.resize_factor;
         camera_matrix.at<double>(2,2) = 1;
      }

      if(!img_1_undist.data || !img_2_undist.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return 1;
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

      } else {
         feat_detector = xfeatures2d::SURF::create(args.detector_data.minHessian, 
               args.detector_data.nOctaves, args.detector_data.nOctaveLayersAkaze, args.detector_data.extended, args.detector_data.upright);
      }

#ifdef TIME
      chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
#endif

      feat_detector->detectAndCompute(img_1_undist, noArray(), KeyPoints_1, descriptors_1);
      feat_detector->detectAndCompute(img_2_undist, noArray(), KeyPoints_2, descriptors_2);

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

      // Convert correspondences to vectors
      vector<Point2f>imgpts1,imgpts2;
      cout << "Number of matches " << matches.size() << endl;
      for(unsigned int i = 0; i < matches.size(); i++) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      /* imgpts1_undist = imgpts1; */
      /* imgpts2_undist = imgpts2; */
      if (args.undistort) 
      {
         undistortPoints(imgpts1, imgpts1_undist, camera_matrix, dist_coefficients, noArray(), camera_matrix);
         undistortPoints(imgpts2, imgpts2_undist, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      } else
      {
         imgpts1_undist = imgpts1;
         imgpts2_undist = imgpts2;
      }
      Mat E = findEssentialMat(imgpts1_undist, imgpts2_undist, 1, Point2d(0,0), RANSAC, 0.999, 8, mask);
      correctMatches(E, imgpts1_undist, imgpts2_undist, imgpts1_undist, imgpts2_undist);

      Mat R, t; // rotation and translation
      cout << "Pose recovery inliers: " << recoverPose(E, imgpts1_undist, imgpts2_undist, R, t, 1.0, Point2d(0,0), mask) << endl;

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

      double err = computeReprojectionError(imgpts1_undist, imgpts2_undist, mask, E);
      cout << "average reprojection err = " <<  err << endl;
      if (args.epilines)
      {
         drawEpilines(Mat(imgpts1_undist), 1, E, img_2_undist);
         drawEpilines(Mat(imgpts2_undist), 2, E, img_1_undist);
      }

      Mat img_matches; // side-by-side comparison
      drawMatches(img_1_undist, KeyPoints_1, img_2_undist, KeyPoints_2, // draw only inliers given by mask
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
