#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

#include "stereo_v3.hpp"

using namespace cv;
using namespace std;

void computePoseDifference(Mat img1, Mat img2, CommandArgs args, Mat camera_matrix, Mat& dist_coefficients, double& worldScale, Mat& R, Mat& t, Mat& img_matches)
{
   cout << "%===============================================%" << endl;
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

   // Find correspondences
   /* BFMatcher matcher(NORM_HAMMING, true); */
   BFMatcher matcher(NORM_HAMMING, false);
   vector<DMatch> matches;

   /* matcher.match(descriptors_1, descriptors_2, matches); */
   vector<vector<DMatch>> match_candidates;
   const float ratio = .9; // Lowe
   matcher.knnMatch(descriptors_1, descriptors_2, match_candidates, 2);
   for (int i = 0; i < match_candidates.size(); i++)
   {
      if (match_candidates[i][0].distance < ratio * match_candidates[i][1].distance)
      {
         matches.push_back(match_candidates[i][0]);
      }
   }

   cout << "Number of matches passing check: " << matches.size() << endl;

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

   /* Why not use these? */
   /* double focal = camera_matrix.at<double>(0,0); */
   /* Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2)); */

   Mat E = findEssentialMat(imgpts1, imgpts2, 1, Point2d(0,0), RANSAC, 0.999, 8, mask);
   correctMatches(E, imgpts1, imgpts2, imgpts1, imgpts2);
   recoverPose(E, imgpts1, imgpts2, R, t, 1.0, Point2d(0,0), mask);

   cout << "Matches used for pose recovery: " << countNonZero(mask) << endl;

   Mat mtxR, mtxQ;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
   cout << "Translation: " << t.t() << endl;
   cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

   if (args.epilines)
   {
      drawEpilines(Mat(imgpts1), 1, E, img2);
      drawEpilines(Mat(imgpts2), 2, E, img1);
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
   worldScale = mDist1;
   cout << "Mean distance of " << n << " points to camera: " << mDist1 << " (dehomogenized)" << endl;
   cout << "pos=" << pos << ", neg=" << neg << endl;

   /* char filename[100]; */
   /* sprintf(filename, "mat_1%d", i+1); */

   /* Ptr<Formatter> formatter = Formatter::get(Formatter::FMT_CSV); */
   /* Ptr<Formatted> formatted = formatter->format(dehomogenized); */
   /* ofstream file(filename, ios_base::trunc); */
   /* file << formatted << endl; */
   cout << "%===============================================%" << endl;
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
