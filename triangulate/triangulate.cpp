#include <opencv2/opencv.hpp>
#include "stereo_v3.hpp"
using namespace std;
using namespace cv;

static int length_filenames = 4;
static char* filenames[] = {
   "/Users/Rasmus/Desktop/Set1/1.jpg",
   "/Users/Rasmus/Desktop/Set1/2.jpg",
   "/Users/Rasmus/Desktop/Set1/3.jpg",
   "/Users/Rasmus/Desktop/Set1/4.jpg"
};
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc,argv);

   for (int i = 1; i < length_filenames; i++) 
   {
      cout << "comparing images " << filenames[0] << " and " << filenames[i] << endl;
      Mat img_1 = imread(filenames[0], IMREAD_COLOR);
      Mat img_2 = imread(filenames[i], IMREAD_COLOR);

      if (!img_1.data || !img_2.data) 
      {
         cerr << "Failed to load." << endl;
         return 1;
      }

      FileStorage fs("../calibration/ipad_camera_params.xml", FileStorage::READ);
      if (fs.isOpened())
      {
         Mat camera_matrix, dist_coefficients;
         fs["Camera_Matrix"] >> camera_matrix;
         fs["Distortion_Coefficients"] >> dist_coefficients;
         fs.release();
         int resize_factor = args.resize_factor;
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
         feat_detector = AKAZE::create(args.detector_data.upright ? AKAZE::DESCRIPTOR_MLDB_UPRIGHT : AKAZE::DESCRIPTOR_MLDB, 
               args.detector_data.descriptor_size,
               args.detector_data.descriptor_channels,
               args.detector_data.threshold,
               args.detector_data.nOctaves,
               args.detector_data.nOctaveLayersAkaze);
         feat_detector->detectAndCompute(img_1, noArray(), KeyPoints_1, descriptors_1);
         feat_detector->detectAndCompute(img_2, noArray(), KeyPoints_2, descriptors_2);

         // Find correspondences
         BFMatcher matcher(NORM_HAMMING, false);
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

         cout << "Number of matches: " << matches.size() << endl;

         // Convert correspondences to vectors
         vector<Point2f> imgpts1, imgpts2;
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
         correctMatches(E, imgpts1_undist, imgpts2_undist, imgpts1_undist, imgpts2_undist);

         Mat R, t; // rotation and translation
         int inliers = recoverPose(E, imgpts1_undist, imgpts2_undist, R, t, 1.0, Point2d(0,0), mask);
         /* cout << "Passing check: " << inliers << " points" << endl; */
         cout << "Translation [x y z]: " << t.t() << endl;

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
         dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols

         double mDist1 = 0;
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
               mDist1 += d;
               n++;
            }
            else neg++;
         }
         mDist1 /= n;
         cout << "Mean distance of " << n << " points to camera: " << mDist1 << " (dehomogenized)" << endl;
         cout << "pos=" << pos << ", neg=" << neg << endl;
      }
   }
   return 0;
}
