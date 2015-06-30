#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, const char *argv[])
{
   Mat img_1 = imread("../stereo/Outdoor Samples/Physik3_+5L.jpg", IMREAD_COLOR);
   Mat img_2 = imread("../stereo/Outdoor Samples/Physik2_+8L2V.jpg", IMREAD_COLOR);

   FileStorage fs("../calibration/ipad_camera_params.xml", FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      int resize_factor = 4;
      resize(img_1, img_1, Size(img_1.cols / resize_factor, img_1.rows / resize_factor)); 
      resize(img_2, img_2, Size(img_2.cols / resize_factor, img_2.rows / resize_factor));
      camera_matrix = camera_matrix / resize_factor;
      camera_matrix.at<double>(2,2) = 1;

      if(!img_1.data || !img_2.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return 1;
      }

      // Feature detection + extraction
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      Mat descriptors_1, descriptors_2;

      Ptr<Feature2D> feat_detector;
      feat_detector = AKAZE::create();
      feat_detector->detectAndCompute(img_1, noArray(), KeyPoints_1, descriptors_1);
      feat_detector->detectAndCompute(img_2, noArray(), KeyPoints_2, descriptors_2);

      BFMatcher matcher(NORM_HAMMING, true);
      vector<DMatch> matches;
      matcher.match(descriptors_1, descriptors_2, matches);
      
      // Convert correspondences to vectors
      vector<Point2f> imgpts1,imgpts2;
      for(unsigned int i = 0; i < matches.size(); i++) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      undistortPoints(imgpts1, imgpts1_undist, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      undistortPoints(imgpts2, imgpts2_undist, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      Mat E = findEssentialMat(imgpts1_undist, imgpts2_undist, 1, Point2d(0,0), RANSAC, 0.999, 8, mask);
      /* correctMatches(E, imgpts1_undist, imgpts2_undist, imgpts1_undist, imgpts2_undist); */

      Mat R, t; // rotation and translation
      int inliers = recoverPose(E, imgpts1_undist, imgpts2_undist, R, t, 1.0, Point2d(0,0), mask);
      /* cout << "Passing check: " << inliers << " points" << endl; */
      /* cout << "Translation [x y z]: " << t.t() << endl; */

      // mask in only those points which were used to compute E
      vector<Point2f> imgpts1_masked, imgpts2_masked;
      for (int i = 0; i < imgpts1_undist.size(); i++) 
      {
         if (mask.at<uchar>(i,0) == 1) 
         {
            imgpts1_masked.push_back(imgpts1_undist[i]);
            imgpts2_masked.push_back(imgpts2_undist[i]);
         }
      }

      Mat pnts4D;
      Mat P1 = Mat::eye(3, 4, CV_64FC1), P2;
      Mat p2[2] = { Mat::eye(3, 3, CV_64FC1), t }; // assume zero rotation until consistent results
      hconcat(p2, 2, P2);
      
      triangulatePoints(P1, P2, imgpts1_masked, imgpts2_masked, pnts4D);
      pnts4D = pnts4D.t();
      Mat dehomogenized;
      convertPointsFromHomogeneous(pnts4D, dehomogenized);
      dehomogenized = dehomogenized.reshape(1); // instead of 3 channels an 1 col, we want 1 channel and 3 cols

      double mDist1 = 0;
      int n = 0;
      Mat_<double> row;

      for (int i = 0; i < dehomogenized.rows; i++) 
      {
         row = dehomogenized.row(i);
         double d = row(2);
         mDist1 += d;
         n++;
      }
      mDist1 /= n;
      cout << "Mean distance of " << n << " points to camera: " << mDist1 << " (dehomogenized)" << endl;
   }
   return 0;
}
