#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "stereo_v3.hpp"
using namespace std;
using namespace cv;

static int length_filenames = 7;
static const char* filenames[] = {
   "/Users/Rasmus/Desktop/Set6/1.jpg",
   "/Users/Rasmus/Desktop/Set6/2.jpg",
   "/Users/Rasmus/Desktop/Set6/3.jpg",
   "/Users/Rasmus/Desktop/Set6/4.jpg",
   "/Users/Rasmus/Desktop/Set6/5.jpg",
   "/Users/Rasmus/Desktop/Set6/6.jpg",
   "/Users/Rasmus/Desktop/Set6/7.jpg",
};
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc,argv);

   for (int i = 1; i < length_filenames; i++) 
   {
      cout << "comparing images " << filenames[0] << " and " << filenames[i] << endl;
      Mat img_1 = imread(filenames[0], IMREAD_COLOR);
      Mat img_2 = imread(filenames[i], IMREAD_COLOR);

      if (!img_1.data || !img_2.data) 
      {
         cerr << "Failed to load." << endl;
         return 1;
      }

      FileStorage fs("../calibration/ipad_camera_params.xml", FileStorage::READ);
      if (fs.isOpened())
      {
         Mat camera_matrix, dist_coefficients;
         fs["Camera_Matrix"] >> camera_matrix;
         fs["Distortion_Coefficients"] >> dist_coefficients;
         fs.release();

         double worldScale;
         Mat R, t, img_matches;
         computePoseDifference(img_1, img_2, args, camera_matrix, dist_coefficients, worldScale, R, t, img_matches);
      }
   }
   return 0;
}
