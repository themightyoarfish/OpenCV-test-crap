/**
 * @file OCV3 compliant version of stereo_v2.cpp
 * @author Rasmus Diederichsen
 */
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas);
double computeReprojectionError(vector<Point2f>& img1pts, vector<Point2f>& img2pts, Mat& inlier_mask, const Mat& F);

int main(int argc, const char *argv[])
{
   if (argc != 4) 
   {
      cout << "Usage: " << argv[0] << " left_img right_img calib_file.xml" << endl;
      return -1;
   }
   Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
   Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

   FileStorage fs(argv[3], FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients, img_1_undist, img_2_undist;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      /* img_1_undist = img_1; */
      /* img_2_undist = img_2; */
      undistort(img_1, img_1_undist, camera_matrix, dist_coefficients); // remove camera imperfections
      undistort(img_2, img_2_undist, camera_matrix, dist_coefficients);
      resize(img_1_undist, img_1_undist, Size(img_1_undist.cols / 5, img_1_undist.rows / 5)); // make smaller for performance and displayablity
      resize(img_2_undist, img_2_undist, Size(img_2_undist.cols / 5, img_2_undist.rows / 5));

      // scale matrix down according to changed resolution
      camera_matrix = camera_matrix / 5.0;
      camera_matrix.at<double>(2,2) = 1;

      if(!img_1_undist.data || !img_2_undist.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return -1;
      }

      // Feature detection + extraction
      int minHessian = 800; // changing this dramatically affects the result, set lower if you want to see nothing 
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      Mat descriptors_1, descriptors_2;

      Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian, 4, 2, true, false);
      surf->detectAndCompute(img_1_undist, noArray(), KeyPoints_1, descriptors_1);
      surf->detectAndCompute(img_2_undist, noArray(), KeyPoints_2, descriptors_2);

      // Find correspondences
      BFMatcher matcher(NORM_L1, true);
      vector<DMatch> matches;
      matcher.match(descriptors_1, descriptors_2, matches);


      // Convert correspondences to vectors
      vector<Point2f>imgpts1,imgpts2;
      for( unsigned int i = 0; i<matches.size(); i++ ) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      imgpts1_undist = imgpts1;
      imgpts2_undist = imgpts2;
      /* undistortPoints(imgpts1, imgpts1_undist, camera_matrix, dist_coefficients); */
      /* undistortPoints(imgpts2, imgpts2_undist, camera_matrix, dist_coefficients); */
      Mat E = findEssentialMat(imgpts1_undist, imgpts2_undist, 1, Point2d(0,0), RANSAC, 0.99, 9, mask);

      Mat R, t; // rotation and translation
      recoverPose(E, imgpts1_undist, imgpts2_undist, R, t);

      double theta_x, theta_y, theta_z;
      theta_x = atan2(R.at<double>(2,1),  R.at<double>(2,2));
      theta_y = atan2(-R.at<double>(2,0), sqrt(pow(R.at<double>(2,1), 2) + pow(R.at<double>(2,2),2)));
      theta_z = atan2(R.at<double>(1,0),  R.at<double>(0,0));

      cout << "Translataion: " << t << endl;

      cout << "\tx rotation: " << theta_x * 180 / M_PI << endl;
      cout << "\ty rotation: " << theta_y * 180 / M_PI << endl;
      cout << "\tz rotation: " << theta_z * 180 / M_PI << endl;

      double err = computeReprojectionError(imgpts1_undist, imgpts2_undist, mask, E);
      cout << "average reprojection err = " <<  err << endl;
      /* drawEpilines(Mat(imgpts1), 1, F, img_2_undist); */
      /* drawEpilines(Mat(imgpts2), 2, F, img_1_undist); */

      Mat img_matches; // side-by-side comparison
      drawMatches(img_1_undist, KeyPoints_1, img_2_undist, KeyPoints_2, // draw only inliers given by mask
            matches, img_matches, Scalar::all(-1), Scalar::all(-1), mask);
      // display
      namedWindow( "Matches", CV_WINDOW_NORMAL );
      imshow("Matches", img_matches );
      waitKey(0);

      return 0;
   }
}


void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas)
{
   // draw the left points corresponding epipolar
   // lines in right image
   vector<Vec3f> lines1;
   computeCorrespondEpilines(
         image_points, // image points
         1, // in image 1 (can also be 2)
         F, // F matrix
         lines1); // vector of epipolar lines
   // for all epipolar lines
   for (vector<Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {
      // draw the line between first and last column
      line(canvas,
            Point(0,-(*it)[2]/(*it)[1]),
            Point(canvas.cols,-((*it)[2]+
                  (*it)[0]*canvas.cols)/(*it)[1]),
            Scalar(255,255,255));
   }
}

double computeReprojectionError(vector<Point2f>& imgpts1, vector<Point2f>& imgpts2, Mat& inlier_mask, const Mat& F)
{
   double err = 0;
   vector<Vec3f> lines[2];
   int npt = sum(inlier_mask)[0]; 

   // strip outliers so validation is constrained to the correspondences
   // which were used to estimate F
   vector<Point2f> imgpts1_copy(npt), 
      imgpts2_copy(npt);
   for (int k = 0; k < inlier_mask.size().height; k++) 
   {
      static int c = 0;
      if (inlier_mask.at<uchar>(0,k) == 1) 
      {
         imgpts1_copy[c] = imgpts1[k];
         imgpts2_copy[c] = imgpts2[k];
         c++;
      } 
   }

   Mat imgpt[2] = { Mat(imgpts1_copy), Mat(imgpts2_copy) };
   computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);
   computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
   for(int j = 0; j < npt; j++ )
   {
      double errij = fabs(imgpts1_copy[j].x*lines[1][j][0] +
            imgpts1_copy[j].y*lines[1][j][1] + lines[1][j][2]) +
         fabs(imgpts2_copy[j].x*lines[0][j][0] +
               imgpts2_copy[j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
   }
   return err / npt;
}
