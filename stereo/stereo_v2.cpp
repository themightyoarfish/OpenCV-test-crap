/**
 * @file This file contains the code compliant with OpenCV2. The program does
 * not compile without cmake due to wtf.
 * @author Rasmus Diederichsen
 */
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"


using namespace cv;
using namespace std;

void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas);
double computeReprojectionError(vector<Point2f>& img1pts, vector<Point2f>& img2pts, Mat& inlier_mask, const Mat& F);
Mat findEssentialMat( InputArray _points1, InputArray _points2, double focal, Point2d pp,
                              int method, double prob, double threshold, OutputArray _mask);

/**
 * @brief Copy-paste from OCV3 code; decomposes essential matrix  into Rotation
 * and translation between two stereo images
 *
 * @param _E Essential matrix (@see findEssenstialMat)
 * @param _R1 One possible rotation
 * @param _R1 Second possible rotation
 * @param _t Unit vector representing direction of translation. The two
 * solutions are _t and -_t
 * @see * http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=recoverpose#decomposeessentialmat
 */
void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t );

/**
 * @brief Copy-paste from OCV3 code
 *
 * @see * http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=recoverpose#recoverpose
 */
int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
      OutputArray _t, double focal, Point2d pp, InputOutputArray _mask);

int main(int argc, const char *argv[])
{
   if (argc != 4) 
   {
      cout << "Usage: " << argv[0] << " left_img right_img calib_file.xml" << endl;
      return -1;
   }
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
      undistort(img_1, img_1_undist, camera_matrix, dist_coefficients); // remove camera imperfections
      undistort(img_2, img_2_undist, camera_matrix, dist_coefficients);
      resize(img_1_undist, img_1_undist, Size(img_1_undist.cols / 5, img_1_undist.rows / 5)); // make smaller for performance and displayablity
      resize(img_2_undist, img_2_undist, Size(img_2_undist.cols / 5, img_2_undist.rows / 5));

      // scale matrix down according to changed resolution
      camera_matrix = camera_matrix / 5.0;
      camera_matrix.at<double>(2,2) = 1;

      if(!img_1_undist.data || !img_2_undist.data) return -1;

      // Feature detection
      int minHessian = 4000; // changing this dramatically affects the result, set lower if you want to see nothing 
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      Mat descriptors_1, descriptors_2;

      SurfFeatureDetector detector(minHessian, 4, 2, true, false);
      detector.detect(img_1_undist, KeyPoints_1);
      detector.detect(img_2_undist, KeyPoints_2);
      
      // Extract descriptors
      SurfDescriptorExtractor extractor;
      extractor.compute(img_1_undist, KeyPoints_1, descriptors_1);
      extractor.compute(img_2_undist, KeyPoints_2, descriptors_2);

      // Find correspondences
      BFMatcher matcher(NORM_L1, true);
      vector<DMatch> matches;
      matcher.match(descriptors_1, descriptors_2, matches);

      // Convert correspondences to vectors
      vector<Point2f>imgpts1,imgpts2;
      for(unsigned int i = 0; i<matches.size(); i++) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      imgpts1_undist = imgpts1; // for now, don't undistort the points
      imgpts2_undist = imgpts2;
      /* undistortPoints(imgpts1, imgpts1_undist, camera_matrix, dist_coefficients); */
      /* undistortPoints(imgpts2, imgpts2_undist, camera_matrix, dist_coefficients); */
      Mat E = findEssentialMat(imgpts1_undist, imgpts2_undist, 1, Point2d(0,0), RANSAC, 0.99, 5, mask);
      recoverPose( E, imgpts1_undist, imgpts2_undist, R, t);

      // decode rotation matrix according to http://nghiaho.com/?page_id=846
      double theta_x, theta_y, theta_z;
      theta_x = atan2(R.at<double>(2,1),  R.at<double>(2,2));
      theta_y = atan2(-R.at<double>(2,0), sqrt(pow(R.at<double>(2,1), 2) + pow(R.at<double>(2,2),2)));
      theta_z = atan2(R.at<double>(1,0),  R.at<double>(0,0));

      cout << "Translataion: " << t << endl;
      cout << "\tx rotation: " << theta_x * 180 / M_PI << endl;
      cout << "\ty rotation: " << theta_y * 180 / M_PI << endl;
      cout << "\tz rotation: " << theta_z * 180 / M_PI << endl;

      double err = computeReprojectionError(imgpts1_undist, imgpts2_undist, mask, E); // compute error
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

void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
{
   Mat E = _E.getMat().reshape(1, 3);
   CV_Assert(E.cols == 3 && E.rows == 3);

   Mat D, U, Vt;
   SVD::compute(E, D, U, Vt);

   if (determinant(U) < 0) U *= -1.;
   if (determinant(Vt) < 0) Vt *= -1.;

   Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
   W.convertTo(W, E.type());

   Mat R1, R2, t;
   R1 = U * W * Vt;
   R2 = U * W.t() * Vt;
   t = U.col(2) * 1.0;

   R1.copyTo(_R1);
   R2.copyTo(_R2);
   t.copyTo(_t);
}
int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
      OutputArray _t, double focal, Point2d pp, InputOutputArray _mask)
{
   Mat points1, points2;
   _points1.getMat().copyTo(points1);
   _points2.getMat().copyTo(points2);

   int npoints = points1.checkVector(2);
   CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
         points1.type() == points2.type());

   if (points1.channels() > 1)
   {
      points1 = points1.reshape(1, npoints);
      points2 = points2.reshape(1, npoints);
   }
   points1.convertTo(points1, CV_64F);
   points2.convertTo(points2, CV_64F);

   points1.col(0) = (points1.col(0) - pp.x) / focal;
   points2.col(0) = (points2.col(0) - pp.x) / focal;
   points1.col(1) = (points1.col(1) - pp.y) / focal;
   points2.col(1) = (points2.col(1) - pp.y) / focal;

   points1 = points1.t();
   points2 = points2.t();

   Mat R1, R2, t;
   decomposeEssentialMat(E, R1, R2, t);
   Mat P0 = Mat::eye(3, 4, R1.type());
   Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
   P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
   P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
   P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
   P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

   // Do the cheirality check.
   // Notice here a threshold dist is used to filter
   // out far away points (i.e. infinite points) since
   // there depth may vary between postive and negtive.
   double dist = 50.0;
   Mat Q;
   triangulatePoints(P0, P1, points1, points2, Q);
   Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
   Q.row(0) /= Q.row(3);
   Q.row(1) /= Q.row(3);
   Q.row(2) /= Q.row(3);
   Q.row(3) /= Q.row(3);
   mask1 = (Q.row(2) < dist) & mask1;
   Q = P1 * Q;
   mask1 = (Q.row(2) > 0) & mask1;
   mask1 = (Q.row(2) < dist) & mask1;

   triangulatePoints(P0, P2, points1, points2, Q);
   Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
   Q.row(0) /= Q.row(3);
   Q.row(1) /= Q.row(3);
   Q.row(2) /= Q.row(3);
   Q.row(3) /= Q.row(3);
   mask2 = (Q.row(2) < dist) & mask2;
   Q = P2 * Q;
   mask2 = (Q.row(2) > 0) & mask2;
   mask2 = (Q.row(2) < dist) & mask2;

   triangulatePoints(P0, P3, points1, points2, Q);
   Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
   Q.row(0) /= Q.row(3);
   Q.row(1) /= Q.row(3);
   Q.row(2) /= Q.row(3);
   Q.row(3) /= Q.row(3);
   mask3 = (Q.row(2) < dist) & mask3;
   Q = P3 * Q;
   mask3 = (Q.row(2) > 0) & mask3;
   mask3 = (Q.row(2) < dist) & mask3;

   triangulatePoints(P0, P4, points1, points2, Q);
   Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
   Q.row(0) /= Q.row(3);
   Q.row(1) /= Q.row(3);
   Q.row(2) /= Q.row(3);
   Q.row(3) /= Q.row(3);
   mask4 = (Q.row(2) < dist) & mask4;
   Q = P4 * Q;
   mask4 = (Q.row(2) > 0) & mask4;
   mask4 = (Q.row(2) < dist) & mask4;

   mask1 = mask1.t();
   mask2 = mask2.t();
   mask3 = mask3.t();
   mask4 = mask4.t();

   // If _mask is given, then use it to filter outliers.
   if (!_mask.empty())
   {
      Mat mask = _mask.getMat();
      CV_Assert(mask.size() == mask1.size());
      bitwise_and(mask, mask1, mask1);
      bitwise_and(mask, mask2, mask2);
      bitwise_and(mask, mask3, mask3);
      bitwise_and(mask, mask4, mask4);
   }
   if (_mask.empty() && _mask.needed())
   {
      _mask.create(mask1.size(), CV_8U);
   }

   CV_Assert(_R.needed() && _t.needed());
   _R.create(3, 3, R1.type());
   _t.create(3, 1, t.type());

   int good1 = countNonZero(mask1);
   int good2 = countNonZero(mask2);
   int good3 = countNonZero(mask3);
   int good4 = countNonZero(mask4);

   if (good1 >= good2 && good1 >= good3 && good1 >= good4)
   {
      R1.copyTo(_R);
      t.copyTo(_t);
      if (_mask.needed()) mask1.copyTo(_mask);
      return good1;
   }
   else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
   {
      R2.copyTo(_R);
      t.copyTo(_t);
      if (_mask.needed()) mask2.copyTo(_mask);
      return good2;
   }
   else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
   {
      t = -t;
      R1.copyTo(_R);
      t.copyTo(_t);
      if (_mask.needed()) mask3.copyTo(_mask);
      return good3;
   }
   else
   {
      t = -t;
      R2.copyTo(_R);
      t.copyTo(_t);
      if (_mask.needed()) mask4.copyTo(_mask);
      return good4;
   }
}
Mat findEssentialMat( InputArray _points1, InputArray _points2, double focal, Point2d pp,
                              int method, double prob, double threshold, OutputArray _mask)
{
    Mat points1, points2;
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 5 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    if( points1.channels() > 1 )
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double ifocal = focal != 0 ? 1./focal : 1.;
    for( int i = 0; i < npoints; i++ )
    {
        points1.at<double>(i, 0) = (points1.at<double>(i, 0) - pp.x)*ifocal;
        points1.at<double>(i, 1) = (points1.at<double>(i, 1) - pp.y)*ifocal;
        points2.at<double>(i, 0) = (points2.at<double>(i, 0) - pp.x)*ifocal;
        points2.at<double>(i, 1) = (points2.at<double>(i, 1) - pp.y)*ifocal;
    }

    // Reshape data to fit opencv ransac function
    points1 = points1.reshape(2, npoints);
    points2 = points2.reshape(2, npoints);

    threshold /= focal;

    Mat E;
    if( method == RANSAC )
        createRANSACPointSetRegistrator(makePtr<EMEstimatorCallback>(), 5, threshold, prob)->run(points1, points2, E, _mask);
    else
        createLMeDSPointSetRegistrator(makePtr<EMEstimatorCallback>(), 5, prob)->run(points1, points2, E, _mask);

    return E;
}
