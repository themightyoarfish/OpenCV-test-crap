/*
 * author: Ann-Katrin Häuser <ahaeuser@uos.de>
 */

#ifndef REPHOTO_SLIDER_H_
#define REPHOTO_SLIDER_H_

#include <opencv2/core.hpp>


/**
* Displays two images on top of each other in a Sliding Window.
* Takes two images of the same scene, and constructs a new image
* by displaying the left part of the old image and the right
* part of the new image. The relation between the sizes of the image
* parts can be dynamically adjustet via a slider.
*/
class Slider {
private:
  struct SlideParams {
    cv::Mat img1;           // first image
    cv::Mat img2;           // second corresponding image
    cv::String window_name; // Name of Displaying Window
  };
  SlideParams params_;

  /**
  * Processes the current trackbarvalue by adjusting the sizes of the
  * displayed images
  *
  * @param pos       Current position of the trackbar
  * @param userdata  All additional data necessary including
  *                  img1, img2 and window_name
  */
  static void trackbarHandler(int pos, void* userdata);

public:
  /**
  * Default Constructor
  */
  Slider();

  /**
  * Set Paramters of Slider Window
  *
  * @param img1         First image displayed on the left of the slider
  * @param img2         Second image displayed on the right
  * @param window_name  Name of the window the images are displayed in
  */
  void setParams(cv::Mat img1, cv::Mat img2,
      cv::String window_name = "ImageSlider");

  /**
  * Displays two corresponding images with a Slider.
  * Via the slider the user is able to adjust the size of the left
  * and right image portion that is displayed
  */
  void displaySliderWindow();
};

#endif  // REPHOTO_SLIDER_H_
