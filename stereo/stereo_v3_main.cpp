#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

#ifdef TIME
#include <chrono>
#endif

#include "stereo_v3.hpp"

int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   if (!args.check_args()) 
   {
      cout << "Usage: " << argv[0] << " --left IMG --right IMG2 --calib CALIB_FILE "
         "[--resize n] [--detector (KAZE|SURF) [--hessianT n] [--octaves n] [--octave-layers n] "
         "[--no-extend] [--upright] [--descriptor-size n] [--descriptor-channels {1,2,3}] [--threshold n]] "
         "[--epilines] [--no-undistort]" << endl;
      return -1;
   }
   Mat img1 = imread(args.left_image_name, IMREAD_COLOR);
   Mat img2 = imread(args.right_image_name, IMREAD_COLOR);

   if(!img1.data || !img2.data) 
   {
      cout << "At least one of the images has no data." << endl;
      return 1;
   }

   FileStorage fs(args.calib_file_name, FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();

      Mat img_matches, R, t;
      double worldScale;
      computePoseDifference(img1, img2, args, camera_matrix, dist_coefficients, worldScale, R, t, img_matches);
      if (args.draw_matches) 
      {
         namedWindow("Matches", CV_WINDOW_NORMAL);
         imshow("Matches", img_matches);
         waitKey(0);
      }
      return 0;
   } else
   {
      cout << "Could not read file " << args.calib_file_name << endl;
      return -1;
   }
}
