#include <opencv2/opencv.hpp>
#include "stereo_v3.hpp"
#include <string>

using namespace std;
using namespace cv;

static vector<string> series = {
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/0_ref.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/1.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/2.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/3.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/4.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/5.JPG",
   "/Users/Rasmus/Desktop/Gut Rosenkrantz/6_first_frame_centered.JPG",
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
      cout << "Preprocessing... " << endl;
      computePoseDifference(firstFrame, secondFrame, args, camera_matrix, dist_coefficients, goalScale, RFirstRef, tFirstRef, img_matches);

      for (int i = 0; i < series.size() - 1; i++) 
      {
         cout << "Comparing image " << series.back() << " and image " << series[i] << endl;
         currentFrame = imread(series[i]);
         double worldScale;
         Mat currentR, currentT;
         computePoseDifference(firstFrame,currentFrame, args, camera_matrix, dist_coefficients, worldScale, currentR, currentT, img_matches);
         cout << "Scale ratio: " <<  goalScale / worldScale << endl;
         Mat necessary_t =  -(-RFirstRef * currentR.t() * currentT + tFirstRef);
         cout << "Necessary translation = " << necessary_t / norm(necessary_t) << endl;
         namedWindow("Matches", CV_WINDOW_NORMAL);
         imshow("Matches", img_matches);
         waitKey(0);
      }
   }
   else cout << "Could not read calibration data." << endl;

   return 0;
}
