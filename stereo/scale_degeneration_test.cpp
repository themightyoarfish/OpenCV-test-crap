#include <opencv2/opencv.hpp>
#include "stereo_v3.hpp"
#include <string>

using namespace std;
using namespace cv;

static vector<string> series = {
   "/Users/Rasmus/Desktop/Set1/1.jpg",
   "/Users/Rasmus/Desktop/Set1/2.jpg",
   "/Users/Rasmus/Desktop/Set1/3.jpg",
   "/Users/Rasmus/Desktop/Set1/4.jpg",
};
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
   Mat firstFrame, secondFrame, reference, currentFrame;
   Mat tFirstRef, RFirstRef, tFirstCurrent, RFirstCurrent;

   firstFrame = imread(series.back()), secondFrame = imread(series.front());
   reference = secondFrame;

   FileStorage fs(args.calib_file_name, FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
   }
   return 0;
}
