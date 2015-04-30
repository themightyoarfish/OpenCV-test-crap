#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas);
double computeReprojectionError(vector<Point2f>& img1pts, vector<Point2f>& img2pts, Mat& inlier_mask, const Mat& F);

int main(int argc, const char *argv[])
{
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );

   FileStorage fs(argv[3], FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients, img_1_undist, img_2_undist;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      /* img_1_undist = img_1; */
      /* img_2_undist = img_2; */
      undistort(img_1, img_1_undist, camera_matrix, dist_coefficients);
      undistort(img_2, img_2_undist, camera_matrix, dist_coefficients);
      resize(img_1_undist, img_1_undist, Size(img_1_undist.cols / 5, img_1_undist.rows / 5));
      resize(img_2_undist, img_2_undist, Size(img_2_undist.cols / 5, img_2_undist.rows / 5));
      camera_matrix = camera_matrix / 5.0;
      camera_matrix.at<double>(2,2) = 1;


      if( !img_1_undist.data || !img_2_undist.data )
      { return -1; }

      //-- Step 1: Detect the KeyPoints using SURF Detector
      int minHessian = 5000;
      SurfFeatureDetector detector( minHessian ,4,2,true,true);
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      detector.detect( img_1_undist, KeyPoints_1 );
      detector.detect( img_2_undist, KeyPoints_2 );

      //-- Step 2: Calculate descriptors (feature vectors)
      SurfDescriptorExtractor extractor;
      Mat descriptors_1, descriptors_2;
      extractor.compute( img_1_undist, KeyPoints_1, descriptors_1 );
      extractor.compute( img_2_undist, KeyPoints_2, descriptors_2 );

      //-- Step 3: Matching descriptor vectors with a brute force matcher
      BFMatcher matcher(NORM_L1, true);
      vector< DMatch > matches;
      matcher.match( descriptors_1, descriptors_2, matches );


      //-- Step 4: calculate Fundamental Matrix
      vector<Point2f>imgpts1,imgpts2;
      for( unsigned int i = 0; i<matches.size(); i++ ) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }
      Mat mask;
      Mat F = findFundamentalMat(imgpts1, imgpts2, CV_FM_RANSAC, 5, 0.99, mask);
      cout << "Number of inliers: " << sum(mask)[0] << endl;
      Mat A = camera_matrix;
      Mat E = F; // we undistorted images above so camera matrix now is unity matrix which means E = F (see Ma et al, ch. 6, p. 178)

      //Perfrom SVD on E
      SVD decomp = SVD(E);
      cout << "W=" << decomp.w << endl;

      //U
      Mat U = decomp.u;

      //S
      Mat S(3, 3, CV_64F, Scalar(0));
      S.at<double>(0, 0) = decomp.w.at<double>(0, 0);
      S.at<double>(1, 1) = decomp.w.at<double>(0, 1);
      S.at<double>(2, 2) = decomp.w.at<double>(0, 2);

      //V
      Mat V = decomp.vt; //Needs to be decomp.vt.t(); (transpose once more)
      Mat T = decomp.u.col(2);
      normalize(T,T);
      cout << "Normalized T = " << T << endl;


      //W
      Mat W(3, 3, CV_64F, Scalar(0));
      W.at<double>(0, 1) = -1;
      W.at<double>(1, 0) = 1;
      W.at<double>(2, 2) = 1;

      cout << "computed rotation: " << endl;
      Mat R = U * W.t() * V;
      double theta_x, theta_y, theta_z;
      theta_x = atan2(R.at<double>(2,1), R.at<double>(2,2));
      theta_y = atan2(-R.at<double>(2,0), sqrt(pow(R.at<double>(2,1), 2) + pow(R.at<double>(2,2),2)));
      theta_z = atan2(R.at<double>(1,0),R.at<double>(0,0));

      cout << "\tx rotation: " << theta_x * 180 / M_PI << endl;
      cout << "\ty rotation: " << theta_y * 180 / M_PI << endl;
      cout << "\tz rotation: " << theta_z * 180 / M_PI << endl;

      double err = computeReprojectionError(imgpts1, imgpts2, mask, F);
      cout << "average reprojection err = " <<  err << endl;
      drawEpilines(Mat(imgpts1), 1, F, img_2_undist);
      drawEpilines(Mat(imgpts2), 2, F, img_1_undist);

      //-- Draw matches
      Mat img_matches;
      drawMatches( img_1_undist, KeyPoints_1, img_2_undist, KeyPoints_2, matches, img_matches );
      //-- Show detected matches
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
   Mat imgpt[2] = { Mat(imgpts1), Mat(imgpts2) };

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

   imgpt[0] = Mat(imgpts1_copy);
   computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);
   imgpt[1] = Mat(imgpts2_copy);
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

