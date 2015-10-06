/*
 * author: Ann-Katrin Häuser <ahaeuser@uos.de>
 *
 * Implements slider.h
 */

#include "manual_registration/slider.h"

#include <iostream>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


/**
* Processes the current trackbarvalue by adjusting the sizes of the
* displayed concat_img
*
* @param pos       Current position of the trackbar
* @param userdata  All additional data necessary including
*                  img1, img2 and window_name
*/
void Slider::trackbarHandler(int pos, void* userdata) {

  SlideParams* data = (SlideParams*)userdata;

  Mat concat_img = Mat(max(data->img1.rows, data->img2.rows),
      max(data->img1.cols, data->img2.cols), CV_8UC3);
  Mat left(concat_img, Rect(0, 0, pos, data->img1.rows));
  Mat right(concat_img, Rect(pos, 0, max(data->img2.cols - pos, 0), data->img2.rows));

  Mat part_img1(data->img1, Rect(0, 0, min(pos, data->img1.cols), data->img1.rows));
  Mat part_img2(data->img2,
      Rect(min(pos, data->img2.cols), 0, max(data->img2.cols - pos, 0) , data->img2.rows));
  part_img1.copyTo(left);
  part_img2.copyTo(right);

  imshow(data->window_name, concat_img);
}


/**
* Default Constructor
*/
Slider::Slider() {
  params_.window_name = "init";
}


/**
* Set Paramters of Slider Window
*
* @param img1         First image displayed on the left of the slider
* @param img2         Second image displayed on the right
* @param window_name  Name of the window the images are displayed in
*/
void Slider::setParams(Mat img1, Mat img2, String window_name) {
  params_.img1 = img1;
  params_.img2 = img2;
  params_.window_name = window_name;
}


/**
* Displays two corresponding images with a Slider.
* Via the slider the user is able to adjust the size of the left
* and right image portion that is displayed
*/
void Slider::displaySliderWindow() {
  if (params_.window_name == "init") {
    cout << "WARNING: Please first set parameters " <<
        "via setParams(img1, img2, ?window_name)" << endl;
  }
  else {
    int trackbarVal = 0;
    int maxVal = max(params_.img1.cols, params_.img2.cols);
    namedWindow(params_.window_name, WINDOW_AUTOSIZE);
    createTrackbar("Slider: ", params_.window_name, &trackbarVal, maxVal,
        Slider::trackbarHandler, &params_);
    imshow(params_.window_name, params_.img2);
    waitKey(0);
    destroyWindow(params_.window_name);
  }
}
