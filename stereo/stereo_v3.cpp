#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

#include "stereo_v3.hpp"


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
