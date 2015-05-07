#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
#include "stereo_v3.hpp"


static ostream& operator<<(ostream& os, const DetectorData& d)
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
static ostream& operator<<(ostream& os, const CommandArgs& d)
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

static CommandArgs parse_args(int& argc, char* const* argv)
{
   CommandArgs args;
   for (int i = 1; i < argc; i++) 
   {
      if (0 == strcmp(argv[i], "--left") && argc > i + 1) 
      {
         args.left_image_name = argv[++i];
      }
      else if (0 == strcmp(argv[i], "--right") && argc > i + 1) 
      {
         args.right_image_name = argv[++i];
      }
      else if (0 == strcmp(argv[i], "--calib") && argc > i + 1) 
      {
         args.calib_file_name = argv[++i];
      }
      else if (0 == strcmp(argv[i], "--resize") && argc > i + 1) 
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
      else if (0 == strcmp(argv[i], "--detector") && argc > i + 1) 
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
      else if (0 == strcmp(argv[i], "--hessianT") && argc > i + 1)
      {
         args.detector_data.minHessian = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--octaves") && argc > i + 1)
      {
         args.detector_data.nOctaves = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--octave-layers") && argc > i + 1)
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
      else if (0 == strcmp(argv[i], "--descriptor-size") && argc > i + 1)
      {
         args.detector_data.descriptor_size = atoi(argv[++i]);
      }
      else if (0 == strcmp(argv[i], "--descriptor-channels") && argc > i + 1)
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

bool check_args(const CommandArgs& args)
{
   if (0 == strcmp(args.left_image_name,"n/a") || 0 == strcmp(args.left_image_name, "n/a") || 0 == strcmp(args.left_image_name, "n/a")) 
      return false;
   else return true;
}

int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   if (!check_args(args)) 
   {
      cout << "Usage: " << argv[0] << " --left IMG --right IMG2 --calib CALIB_FILE "
         "[--resize n] [--detector (KAZE|SURF) [--hessianT n] [--octaves n] [--octave-layers = n] "
         "[--no-extend] [--upright] [--descriptor-size n] [--descriptor-channels {1,2,3}] [--threshold n]] "
         "[--epilines] [--undistort] [--no-undistort]" << endl;
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
      if (args.undistort) 
      {
         undistort(img_1, img_1_undist, camera_matrix, dist_coefficients); // remove camera imperfections
         undistort(img_2, img_2_undist, camera_matrix, dist_coefficients);
      } else
      {
         img_1_undist = img_1;
         img_2_undist = img_2;
      }
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
         return -1;
      }

      // Feature detection + extraction
      vector<KeyPoint> KeyPoints_1, KeyPoints_2;
      Mat descriptors_1, descriptors_2;

      Ptr<Feature2D> feat_detector;
      if (args.detector == DETECTOR_KAZE) 
      {
      feat_detector = AKAZE::create(args.detector_data.upright ? AKAZE::DESCRIPTOR_KAZE_UPRIGHT : AKAZE::DESCRIPTOR_KAZE, 
            args.detector_data.descriptor_size,
            args.detector_data.descriptor_channels,
            args.detector_data.threshold,
            args.detector_data.nOctaves,
            args.detector_data.nOctaveLayersAkaze);
         
      } else {
         feat_detector = xfeatures2d::SURF::create(args.detector_data.minHessian, 
               args.detector_data.nOctaves, args.detector_data.nOctaveLayersAkaze, args.detector_data.extended, args.detector_data.upright);
      }
      feat_detector->detectAndCompute(img_1_undist, noArray(), KeyPoints_1, descriptors_1);
      feat_detector->detectAndCompute(img_2_undist, noArray(), KeyPoints_2, descriptors_2);

      // Find correspondences
      BFMatcher matcher(NORM_L2, true);
      vector<DMatch> matches;
      matcher.match(descriptors_1, descriptors_2, matches);


      // Convert correspondences to vectors
      vector<Point2f>imgpts1,imgpts2;
      cout << "Number of matches " << matches.size() << endl;
      for(unsigned int i = 0; i<matches.size(); i++) 
      {
         imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
         imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
      }

      Mat mask; // inlier mask
      vector<Point2f> imgpts1_undist, imgpts2_undist;
      imgpts1_undist = imgpts1;
      imgpts2_undist = imgpts2;
      /* undistortPoints(imgpts1, imgpts1_undist, camera_matrix, dist_coefficients); */ // this doesn't work
      /* undistortPoints(imgpts2, imgpts2_undist, camera_matrix, dist_coefficients); */
      Mat E = findEssentialMat(imgpts1_undist, imgpts2_undist, 1, Point2d(0,0), RANSAC, 0.999, 8, mask);
      /* correctMatches(E, imgpts1_undist, imgpts2_undist, imgpts1_undist, imgpts2_undist); */

      Mat R, t; // rotation and translation
      cout << "Pose recovery inliers: " << recoverPose(E, imgpts1_undist, imgpts2_undist, R, t) << endl;

      double theta_x, theta_y, theta_z;
      theta_x = atan2(R.at<double>(2,1),  R.at<double>(2,2));
      theta_y = atan2(-R.at<double>(2,0), sqrt(pow(R.at<double>(2,1), 2) + pow(R.at<double>(2,2),2)));
      theta_z = atan2(R.at<double>(1,0),  R.at<double>(0,0));

      cout << "Translation: " << t << endl;

      cout << "\tx rotation: " << theta_x * 180 / M_PI << endl;
      cout << "\ty rotation: " << theta_y * 180 / M_PI << endl;
      cout << "\tz rotation: " << theta_z * 180 / M_PI << endl;

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


static void drawEpilines(const Mat& image_points, int whichImage, Mat& F, Mat& canvas)
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

static double computeReprojectionError(vector<Point2f>& imgpts1, vector<Point2f>& imgpts2, Mat& inlier_mask, const Mat& F)
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
