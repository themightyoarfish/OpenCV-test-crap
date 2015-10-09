#include "DualImageWindow.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

static const int ESC = 27; // ascii of escape key

namespace imagelabeling 
{
   using namespace cv;
   using std::cout;
   using std::endl;
   DualImageWindow::DualImageWindow(const Mat left_img, const Mat right_img, const string window_name)
   {
      this->left_img = left_img;
      this->right_img = right_img;
      this->window_name = window_name;
      firstPointSet = false;
   }
   DualImageWindow::~DualImageWindow()
   {
      destroyWindow(window_name);
   }
   void DualImageWindow::combine_imgs(const Mat& left_img, const Mat& right_img)
   {
      combined_imgs = Mat(max(left_img.rows, right_img.rows), left_img.cols + right_img.cols, left_img.type());
      Mat left_roi(combined_imgs, Rect(0, 0, left_img.cols, left_img.rows));
      Mat right_roi(combined_imgs, Rect(left_img.cols, 0, right_img.cols, right_img.rows));
      left_img.copyTo(left_roi);
      right_img.copyTo(right_roi);
   }
   void DualImageWindow::refresh()
   {
      combine_imgs(left_img, right_img);
      for (auto iter = correspondences.begin() ; iter != correspondences.end() ; iter++)
         circle(combined_imgs, *iter, 8, Scalar(0,0,255), 2);
      imshow(window_name, combined_imgs);
   }
   void DualImageWindow::show()
   {
      combine_imgs(left_img, right_img);

      namedWindow(window_name, WINDOW_NORMAL);
      refresh();
      setMouseCallback(window_name, mouseCallback, this);
      while(handleKeyEvent(waitKey(0)));

   }
   bool DualImageWindow::handleKeyEvent(const int key) const
   {
      if(key == ESC) return false;
      else
      {
         return true;
      }
   }
   void DualImageWindow::mouseCallback(int event, int x, int y, int flags, void* data)
   {

      if (data) 
      {
         DualImageWindow* self = static_cast<DualImageWindow*>(data);
         switch (event)
         {
            case EVENT_LBUTTONDOWN: 
               if (flags & EVENT_FLAG_SHIFTKEY) // use the KEY as a bitmask. This will break if they ever change the enum values.
               {
                  if (self->firstPointSet) 
                  {
                     if (x >= self->left_img.cols) 
                     {
                        self->correspondences.push_back(Point2i(x,y));
                        cout << "Circle at " << Point2i(x,y) << endl;
                        self->firstPointSet = false;
                     }
                  } else 
                  {
                     if (x < self->left_img.cols) 
                     {
                        self->correspondences.push_back(Point2i(x,y));
                        cout << "Circle at " << Point2i(x,y) << endl;
                        self->firstPointSet = true;
                     }
                  }
                  self->refresh();
               }
               break;
            default:
               break;
         }

      }
   }
} /* namespace imagelabeling */

