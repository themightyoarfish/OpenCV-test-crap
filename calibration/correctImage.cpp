#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char *argv[])
{
   using namespace cv;
   using namespace std;

   if (argc < 4) 
   {
      cout << "usage: " << argv[0] << " { image } { parameter file } { output file }" << endl;
      return -1;
   } else
   {
      cout << "Correcting image " << argv[1] << " with calibration data " << argv[2] << " ..." << endl;
      FileStorage fs(argv[2], FileStorage::READ);
      if (fs.isOpened())
      {
         Mat original = imread(argv[1]);
         Mat camera_matrix, dist_coefficients, undistorted, undistorted_small;
         fs["Camera_Matrix"] >> camera_matrix;
         fs["Distortion_Coefficients"] >> dist_coefficients;
         fs.release();
         if (!camera_matrix.empty() && !dist_coefficients.empty()) 
         {
            // beware that image orientation is not preserved
            undistort(original, undistorted, camera_matrix, dist_coefficients);
#ifdef SHOW_IMGS
            cout << "Showing undistorted image ..." << endl;
            namedWindow("Original", WINDOW_AUTOSIZE);
            resize(original, original, Size(original.cols / 4, original.rows / 4));
            imshow("Original", original);
            namedWindow("Undistorted", WINDOW_AUTOSIZE);
            resize(undistorted, undistorted_small, Size(undistorted.cols / 4, undistorted.rows / 4));
            imshow("Undistorted", undistorted_small);
            waitKey(0); 
#endif
            cout << "Saving undistorted image ..." << endl;
            imwrite(argv[3], undistorted);

         }
         cout << "Finished." << endl;
         return 0;
      } else
      {
         cout << "Could not open " << argv[2] << " for reading." << endl;
         return -1;
      }
   }
}
