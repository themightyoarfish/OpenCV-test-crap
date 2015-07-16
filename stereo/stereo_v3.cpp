#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include "Common.h"
#include "Triangulation.h"

#include "stereo_v3.hpp"

using namespace cv;
using namespace std;

void computePoseDifference(Mat img1, Mat img2, CommandArgs args, Mat k, Mat& dist_coefficients, double& worldScale, Mat& R, Mat& t, Mat& img_matches)
{
   cout << "%===============================================%" << endl;

   Mat camera_matrix = k.clone();
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
      feat_detector = xfeatures2d::SURF::create(args.detector_data.minHessian, 
            args.detector_data.nOctaves, args.detector_data.nOctaveLayersAkaze, args.detector_data.extended, args.detector_data.upright);

   feat_detector->detectAndCompute(img1, noArray(), KeyPoints_1, descriptors_1);
   feat_detector->detectAndCompute(img2, noArray(), KeyPoints_2, descriptors_2);

   cout << "Number of feature points (img1, img2): " << "(" << KeyPoints_1.size() << ", " << KeyPoints_2.size() << ")" << endl;

   // Find correspondences
   BFMatcher matcher;
   vector<DMatch> matches;
   if (args.use_ratio_test) 
   {
      if (args.detector == DETECTOR_KAZE) 
         matcher = BFMatcher(NORM_HAMMING, false);
      else matcher = BFMatcher(NORM_L2, false);

      vector<vector<DMatch>> match_candidates;
      const float ratio = args.ratio;
      matcher.knnMatch(descriptors_1, descriptors_2, match_candidates, 2);
      for (int i = 0; i < match_candidates.size(); i++)
         if (match_candidates[i][0].distance < ratio * match_candidates[i][1].distance)
            matches.push_back(match_candidates[i][0]);

      cout << "Number of matches passing ratio test: " << matches.size() << endl;

   } else
   {
      if (args.detector == DETECTOR_KAZE) 
         matcher = BFMatcher(NORM_HAMMING, true);
      else matcher = BFMatcher(NORM_L2, true);
      matcher.match(descriptors_1, descriptors_2, matches);
      cout << "Number of matching feature points: " << matches.size() << endl;
   }


   // Convert correspondences to vectors
   vector<Point2f>imgpts1,imgpts2;

   for(unsigned int i = 0; i < matches.size(); i++) 
   {
      imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
      imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
   }

   Mat mask; // inlier mask
   if (args.undistort) 
   {
      undistortPoints(imgpts1, imgpts1, K1, dist_coefficients, noArray(), K1);
      undistortPoints(imgpts2, imgpts2, K2, dist_coefficients, noArray(), K2);
   } 

   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat E = findEssentialMat(imgpts1, imgpts2, focal, principalPoint, RANSAC, 0.999, 1, mask);
   /* Mat F = camera_matrix.t().inv() * E * camera_matrix.inv(); */
   Mat F = findFundamentalMat(imgpts1, imgpts2, CV_FM_RANSAC);

   correctMatches(F, imgpts1, imgpts2, imgpts1, imgpts2);
   cout << "Reprojection error: " << computeReprojectionError(imgpts1, imgpts2, mask, F) << endl;

   int inliers = recoverPose(E, imgpts1, imgpts2, R, t, focal, principalPoint, mask);

   cout << "Matches used for pose recovery: " << inliers << endl;
   
   /* Mat R1, R2, ProjMat1, ProjMat2, Q; */
   /* stereoRectify(camera_matrix, dist_coefficients, camera_matrix, dist_coefficients, img1.size(), R, t, R1, R2, ProjMat1, ProjMat2, Q); */
   /* cout << "P1=" << ProjMat1 << endl; */
   /* cout << "P2=" << ProjMat2 << endl; */
   /* cout << "Q=" << Q << endl; */

   Mat mtxR, mtxQ;
   Mat Qx, Qy, Qz;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ, Qx, Qy, Qz);
   cout << "Qx: " << Qx << endl;
   cout << "Qy: " << Qy << endl;
   cout << "Qz: " << Qz << endl;
   cout << "Translation: " << t.t() << endl;
   cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

   if (args.epilines)
   {
      drawEpilines(Mat(imgpts1), 1, F, img2);
      drawEpilines(Mat(imgpts2), 2, F, img1);
   }

   drawMatches(img1, KeyPoints_1, img2, KeyPoints_2, // draw only inliers given by mask
         matches, img_matches, Scalar::all(-1), Scalar::all(-1), mask);

   vector<Point2f> imgpts1_masked, imgpts2_masked;
   for (int i = 0; i < imgpts1.size(); i++) 
   {
      if (mask.at<uchar>(i,0) == 1) 
      {
         imgpts1_masked.push_back(imgpts1[i]);
         imgpts2_masked.push_back(imgpts2[i]);
      }
   }

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;
   
#define USE_OPENCV_TRIANGULATION
#ifndef USE_OPENCV_TRIANGULATION // strangely, both methods yield identical results
   vector<Point3d> homogPoints1, homogPoints2;
   for (int i = 0; i < imgpts1_masked.size(); i++) 
   {
      Point2f currentPoint1 = imgpts1_masked[i];
      homogPoints1.push_back(Point3d(currentPoint1.x, currentPoint1.y, 1));
      Point2f currentPoint2 = imgpts2_masked[i];
      homogPoints2.push_back(Point3d(currentPoint2.x, currentPoint2.y, 1));
   }

   Mat dehomogenized(imgpts1_masked.size(), 3, CV_64FC1);
   for (int i = 0; i < imgpts1_masked.size(); i++) 
   {
      Mat_<double> triangulatedPoint = IterativeLinearLSTriangulation(homogPoints1[i], P1, homogPoints2[i], P2);
      Mat r = triangulatedPoint.t();
      r.colRange(0,3).copyTo(dehomogenized.row(i)); // directly assigning to dehomogenized.row(i) compiles but does nothing, wtf?
   }
#else
   triangulatePoints(P1, P2, imgpts1_masked, imgpts2_masked, pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols
#endif


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
         mDist += d;
         n++;
         /* float startx=imgpts1_masked[i].x - 1, starty=imgpts1_masked[i].y - 1, endx=imgpts1_masked[i].x + 1, endy=imgpts1_masked[i].y + 1; */
         /* cout << "startx,endx = " << startx << "," << endx << endl; */
         /* cout << "starty,endy = " << starty << "," << endy << endl; */
         Vec3b rgb = img1.at<Vec3b>(imgpts1_masked[i].x, imgpts1_masked[i].y);
         ply_file << row(0) << " " << row(1) << " " << row(2) << " " << (int)rgb[2] << " " << (int)rgb[1] << " " << (int)rgb[0] << "\n";
      } else
      {
         neg++;
         ply_file << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << "\n"; 
      }
   }
   ply_file.close();
   mDist /= n;
   worldScale = mDist;
   cout << "Mean distance of " << n << " points to camera: " << mDist << " (dehomogenized)" << endl;
   cout << "pos=" << pos << ", neg=" << neg << endl;


   /* char filename[100]; */
   /* sprintf(filename, "mat_1%d", i+1); */

   /* Ptr<Formatter> formatter = Formatter::get(Formatter::FMT_CSV); */
   /* Ptr<Formatted> formatted = formatter->format(dehomogenized); */
   /* ofstream file(filename, ios_base::trunc); */
   /* file << formatted << endl; */

   /* Removed until cmake has been fathomed */
   /* vector< Point3d > points3D; */
   /* vector< vector< Point2d > > pointsImg; */
   /* int NPOINTS=dehomogenized.rows; // number of 3d points */
   /* int NCAMS=2; // number of cameras */

   /* points3D.resize(NPOINTS); */
   /* for (int i = 0; i < NPOINTS; i++) */ 
   /* { */
   /*    points3D[i] = Point3d(dehomogenized.at<double>(i,0), */
   /*          dehomogenized.at<double>(i,1), */
   /*          dehomogenized.at<double>(i,2) */
   /*          ); */
   /* } */
   /* // fill image projections */
   /* vector<vector<int> > visibility(2, vector<int>(NPOINTS, 1)); */
   /* vector<Mat> camera_matrices(2, camera_matrix); */
   /* vector<Mat> Rs(2); */
   /* Rodrigues(Mat::eye(3, 3, CV_64FC1), Rs[0]); */
   /* Rodrigues(R, Rs[0]); */
   /* vector<Mat> Ts = { Mat::zeros(3,1, CV_64FC1), t }; */
   /* vector<Mat> dist_coefficientss(2, dist_coefficients); */

   /* pointsImg.resize(NCAMS); */
   /* for(int i=0; i<NCAMS; i++) pointsImg[i].resize(NPOINTS); */
   /* for (int i = 0; i < NPOINTS; i++) */ 
   /* { */
   /*    pointsImg[0][i] = Point2d(imgpts1_masked[i].x, imgpts1_masked[i].y); */
   /*    pointsImg[1][i] = Point2d(imgpts2_masked[i].x, imgpts2_masked[i].y); */
   /* } */
   /*  cvsba::Sba sba; */
   /*   sba.run(points3D, pointsImg, visibility, camera_matrices, Rs, Ts, dist_coefficientss); */

   /*   cout<<"Initial error="<<sba.getInitialReprjError()<<". "<< */
   /*              "Final error="<<sba.getFinalReprjError()<<endl; */

   cout << "%===============================================%" << endl;
}

void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas)
{
   // draw the left points corresponding epipolar
   // lines in right image
   vector<Vec3f> lines1;
   computeCorrespondEpilines(
         image_points, // image points
         whichImage, // in image 1
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
            Scalar(0,0,255), // red
            1);
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
   int c = 0;
   for (int k = 0; k < inlier_mask.size().height; k++) 
   {
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
        // error is computed as the distance between a point u_l = (x,y) and the epipolar line of its corresponding point u_r in the second image plus the reverse, so errij = d(u_l, F^T * u_r) + d(u_r, F*u_l)
        Point2f u_l = imgpts1_copy[j], // for the purpose of this function, we imagine imgpts1 to be the "left" image and imgpts2 the "right" one. Doesn't make a difference
                u_r = imgpts2_copy[j];
        float a2 = lines[1][j][0], // epipolar line
              b2 = lines[1][j][1],
              c2 = lines[1][j][2];
        float norm_factor2 = sqrt(pow(a2, 2) + pow(b2, 2));
        float a1 = lines[0][j][0],
              b1 = lines[0][j][1],
              c1 = lines[0][j][2];
        float norm_factor1 = sqrt(pow(a1, 2) + pow(b1, 2));
        
        double errij =
        fabs(u_l.x * a2 + u_l.y * b2 + c2) / norm_factor2 +
        fabs(u_r.x * a1 + u_r.y * b1 + c1) / norm_factor1; // distance of (x,y) to line (a,b,c) = ax + by + c / (a^2 + b^2)
        err += errij;
    }

   return err / npt;
}

ostream& operator<<(ostream& os, const DetectorData& d)
{
   os << boolalpha;
   os << "Detector data: " << "\n"
      << "\tminHessian: " << d.minHessian << "\n"
      << "\tnOctaves: " << d.nOctaves << "\n"
      << "\tnOctaveLayersSurf: " << d.nOctaveLayersSurf << "\n"
      << "\tnOctaveLayersAkaze: " << d.nOctaveLayersAkaze << "\n"
      << "\textended: " << d.extended << "\n"
      << "\tupright: " << d.upright << "\n"
      << "\tDescriptor size: " << d.descriptor_size << "\n"
      << "\tDescriptor channels: " << d.descriptor_channels << "\n"
      << "\tthreshold: " << d.threshold << "\n";
   return os;
}
ostream& operator<<(ostream& os, const CommandArgs& d)
{
   os << boolalpha;
   os << "Left image: " << d.left_image_name << "\n"
      << "Right image: " << d.right_image_name << "\n"
      << "Calib file: " << d.calib_file_name << "\n"
      << "Resize factor: " << d.resize_factor << "\n"
      << "Undistort: " << d.undistort << "\n"
      << "Epilines: " << d.epilines << "\n"
      << "Detector type: " << (d.detector == DETECTOR_SURF ? "SURF" : "KAZE") << "\n"
      << d.detector_data;
   return os;
}

#define IS_ARG(vec,param) ((0 == strcmp(vec,param)) && (argc > i + 1))

CommandArgs parse_args(int& argc, char* const* argv)
{
   CommandArgs args;
   for (int i = 1; i < argc; i++) 
   {
      if (IS_ARG(argv[i], "--left")) 
      {
         args.left_image_name = argv[++i];
      }
      else if (0 == strcmp(argv[i], "--draw-matches"))
      {
         args.draw_matches = true;
      }
      else if (0 == strcmp(argv[i], "--ratioTest"))
      {
         args.use_ratio_test = true;
      }
      else if (IS_ARG(argv[i], "--ratio"))
      {
         args.ratio = atof(argv[++i]);
      }
      else if (IS_ARG(argv[i], "--right")) 
      {
         args.right_image_name = argv[++i];
      }
      else if (IS_ARG(argv[i], "--calib")) 
      {
         args.calib_file_name = argv[++i];
      }
      else if (IS_ARG(argv[i], "--resize")) 
      {
         args.resize_factor = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--no-undistort")) 
      {
         args.undistort = false;
      }
      else if (0 == strcmp(argv[i], "--epilines")) 
      {
         args.epilines = true;
      }
      else if (IS_ARG(argv[i], "--detector")) 
      { 
         if (0 == strcmp(argv[i+1], "KAZE")) 
         {
            args.detector = DETECTOR_KAZE;
         }
         else if (0 == strcmp(argv[i+1], "SURF")) 
         {
            args.detector = DETECTOR_SURF;
         }
         else cout << "Unknonw detector " << argv[i+1] << endl;
         i++;
      }
      else if (IS_ARG(argv[i], "--hessianT"))
      {
         args.detector_data.minHessian = atoi(argv[++i]);
      }
      else if (IS_ARG(argv[i], "--octaves"))
      {
         args.detector_data.nOctaves = atoi(argv[++i]);
      }
      else if (IS_ARG(argv[i], "--octave-layers"))
      {
         args.detector_data.nOctaveLayersSurf = args.detector_data.nOctaveLayersAkaze = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--no-extended"))
      {
         args.detector_data.extended = false;
      }
      else if (0 == strcmp(argv[i], "--upright"))
      {
         args.detector_data.upright = true;
      }
      else if (IS_ARG(argv[i], "--descriptor-size"))
      {
         args.detector_data.descriptor_size = atoi(argv[++i]);
      }
      else if (IS_ARG(argv[i], "--descriptor-channels"))
      {
         args.detector_data.descriptor_channels = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--threshold") && (argc > i + 1))
      {
         args.detector_data.threshold = static_cast<float>(atof(argv[++i]));
      }
      else cout << "Useless parameter: " << argv[i++] << endl;
   }
   return args;
}
