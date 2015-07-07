#include <opencv2/opencv.hpp>
#include "stereo_v3.hpp"
#include <string>

using namespace std;
using namespace cv;

static vector<string> series = {
   "/Users/Rasmus/Desktop/Set5/1.jpg",
   "/Users/Rasmus/Desktop/Set5/2.jpg",
   "/Users/Rasmus/Desktop/Set5/3.jpg",
   "/Users/Rasmus/Desktop/Set5/4.jpg",
};
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);

   Mat firstFrame, secondFrame, reference, currentFrame;
   Mat tFirstRef, RFirstRef, tFirstCurrent, RFirstCurrent;
   double goalScale;

   firstFrame = imread(series.back()), secondFrame = imread(series.front());
   reference = secondFrame;

   FileStorage fs(args.calib_file_name, FileStorage::READ);
   if (fs.isOpened())
   {
      Mat camera_matrix, dist_coefficients;
      fs["Camera_Matrix"] >> camera_matrix;
      fs["Distortion_Coefficients"] >> dist_coefficients;
      fs.release();
      
      // preprocessing
      Mat img_matches; // unused
      cout << "Preprocessing... (image 4 and image 1)" << endl;
      computePoseDifference(firstFrame, secondFrame, args, camera_matrix, dist_coefficients, goalScale, RFirstRef, tFirstRef, img_matches);

      for (int i = 0; i < series.size() - 1; i++) 
      {
         cout << "Comparing image 4 and image " << i+1 << endl;
         currentFrame = imread(series[i]);
         double worldScale;
         Mat currentR, currentT;
         computePoseDifference(firstFrame,currentFrame, args, camera_matrix, dist_coefficients, worldScale, currentR, currentT, img_matches);
         cout << "Scale ratio: " <<  goalScale / worldScale << endl;
      }
   }

   return 0;
}
