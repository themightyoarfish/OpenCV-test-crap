#include "manual_registration/slider.h"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/* Usage Instructions
*/
void readme() {
  cout << " Usage: ./DiplayImageSlider <img1> <img2>" << endl;
  cout << "        ./DiplayImageSlider <img1> <img2> <yaml>" << endl;
}

int main(int argc, char** argv) {
  Mat H, img1, img2, warp_img;
  Slider slider;
  FileStorage fs;

  //Check number of given params
  switch (argc-1) {
    case 2:
      img1 = imread(argv[1], 1);
      img2 = imread(argv[2], 1);
      if (!img1.data || !img2.data) {
        cout << " --(!) Unable to read at least one of these images: " << endl;
      }
      else {
        slider.setParams(img1, img2);
        slider.displaySliderWindow();
      }
      break;
    case 3:
      img1 = imread(argv[1], 1);
      img2 = imread(argv[2], 1);
      if (!img1.data || !img2.data) {
        cout << " --(!) Unable to read at least one of these images: " << endl;
      }
      else {
        fs.open(argv[3], FileStorage::READ);
        fs["Homography"] >> H;
        warpPerspective(img1, warp_img, H, img1.size());
        slider.setParams(warp_img, img2);
        slider.displaySliderWindow();
      }
      break;

    default:
      readme();
  }
  return 0;
}
