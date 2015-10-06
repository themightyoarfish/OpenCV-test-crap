#include "DualImageWindow.hpp"
#include <opencv2/highgui.hpp>

namespace imagelabeling 
{
   using namespace cv;
   DualImageWindow::DualImageWindow(Mat left_img, Mat right_img, string window_name)
   {
      this->left_img = left_img;
      this->right_img = right_img;
      this->window_name = window_name;
   }
   DualImageWindow::~DualImageWindow()
   {
      destroyWindow(window_name);
   }
   void DualImageWindow::show()
   {

      Mat combined_imgs(max(left_img.rows, right_img.rows), left_img.cols + right_img.cols, left_img.type());
      Mat left_roi(combined_imgs, Rect(0, 0, left_img.cols, left_img.rows));
      Mat right_roi(combined_imgs, Rect(left_img.cols, 0, right_img.cols, right_img.rows));
      left_img.copyTo(left_roi);
      right_img.copyTo(right_roi);

      namedWindow(window_name, WINDOW_NORMAL);
      imshow(window_name, combined_imgs);
      waitKey(0);

   }
} /* namespace imagelabeling */

