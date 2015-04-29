#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>

int main(int argc, const char *argv[])
{
   using namespace cv;
   using namespace std;
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );

   FileStorage fs(argv[3], FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients, img_1_undist, img_2_undist;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      undistort(img_1, img_1_undist, camera_matrix, dist_coefficients);
      undistort(img_2, img_2_undist, camera_matrix, dist_coefficients);
      resize(img_1_undist, img_1_undist, Size(img_1_undist.cols / 5, img_1_undist.rows / 5));
      resize(img_2_undist, img_2_undist, Size(img_2_undist.cols / 5, img_2_undist.rows / 5));


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
      Mat mask(imgpts1.size(), 1, CV_8UC1);
      Mat F = findFundamentalMat(imgpts1, imgpts2, CV_FM_RANSAC, 0.1, 0.99, mask);
      /* for (vector<int>::iterator it = mask.begin(); it != mask.end(); it++) */ 
      /* { */
      /*    cout << *it << endl; */
      /* } */
      Mat A = camera_matrix;
      Mat E = A.t() * F * A;

      //Perfrom SVD on E
      SVD decomp = SVD(E);

      //U
      Mat U = decomp.u;

      //S
      Mat S(3, 3, CV_64F, Scalar(0));
      S.at<double>(0, 0) = decomp.w.at<double>(0, 0);
      S.at<double>(1, 1) = decomp.w.at<double>(0, 1);
      S.at<double>(2, 2) = decomp.w.at<double>(0, 2);

      //V
      Mat V = decomp.vt; //Needs to be decomp.vt.t(); (transpose once more)

      //W
      Mat W(3, 3, CV_64F, Scalar(0));
      W.at<double>(0, 1) = -1;
      W.at<double>(1, 0) = 1;
      W.at<double>(2, 2) = 1;

      cout << "computed rotation: " << endl;
      cout << U * W.t() * V.t() << endl;

      double err = 0;
      int npoints = 0;
      vector<Vec3f> lines[2];
      int npt = (int)imgpts1.size();
      Mat imgpt[2];
      for(int k = 0; k < 2; k++ )
      {
         switch(k)
         {
            case 0:
               imgpt[k] = Mat(imgpts1);
               computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
               break;
            case 1:
               imgpt[k] = Mat(imgpts2);
               computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
               break;
         }
      }
      for(int j = 0; j < npt; j++ )
      {
         double errij = fabs(imgpts1[j].x*lines[1][j][0] +
               imgpts1[j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(imgpts2[j].x*lines[0][j][0] +
                  imgpts2[j].y*lines[0][j][1] + lines[0][j][2]);
         err += errij;
      }
      npoints += npt;
      cout << "average reprojection err = " <<  err/npoints << endl;

      // draw the left points corresponding epipolar
      // lines in right image
      vector<Vec3f> lines1;
      computeCorrespondEpilines(
            Mat(imgpts1), // image points
            1, // in image 1 (can also be 2)
            F, // F matrix
            lines1); // vector of epipolar lines
      // for all epipolar lines
      for (vector<Vec3f>::const_iterator it= lines1.begin();
            it!=lines1.end(); ++it) {
         // draw the line between first and last column
         line(img_2_undist,
               Point(0,-(*it)[2]/(*it)[1]),
               Point(img_2_undist.cols,-((*it)[2]+
                     (*it)[0]*img_2_undist.cols)/(*it)[1]),
               Scalar(255,255,255));
      }
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


