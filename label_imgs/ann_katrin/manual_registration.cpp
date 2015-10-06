/*
 * author: Ann-Katrin Häuser <ahaeuser@uos.de>
 *
 * Implements manual_registration.h
 */

#include "manual_registration/manual_registration.h"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;


/**
* Displays image pair next to each other and saves it in image_pair_
*/
void ManualRegistration::displayImagesSideBySide() {
  img_pair_ = Mat(max(img1_.rows, img2_.rows), img1_.cols + img2_.cols,
      CV_8UC3);
  Mat left(img_pair_, Rect(0, 0, img1_.cols, img1_.rows));
  Mat right(img_pair_, Rect(img1_.cols, 0, img2_.cols, img2_.rows));
  img1_.copyTo(left);
  img2_.copyTo(right);
  imshow(window_name_, img_pair_);
}


/**
* Draws all previously selected points on top of image_pair_
*/
void ManualRegistration::drawPoints(void* userdata) {
  RegistParams* data = (RegistParams*)userdata;

  Mat marked_img_pair = data->img_pair.clone();
  for (int i = 0; i < data->points_img1.size(); i++) {
    circle(marked_img_pair, data->points_img1[i], 4, data->color[i % 8], 2);
  }
  for (int i = 0; i < data->points_img2.size(); i++) {
    circle(marked_img_pair,
        Point(data->points_img2[i].x + data->cols_img1, data->points_img2[i].y),
        4, data->color[i % 8], 2);
  }
  imshow(data->window_name, marked_img_pair);
}


/**
* Handles LeftMouseButtonClicks to mark corresponding points images.
* The last selection is deleted upon additional press of the SHIFTKEY
*/
void ManualRegistration::mouseHandler(int event, int x, int y, int flags,
    void* userdata) {
  RegistParams* data = (RegistParams*)userdata;

  switch (event) {
    case EVENT_LBUTTONUP:
      if (flags & EVENT_FLAG_SHIFTKEY) { // delete point
        if (x < data->cols_img1) {
          data->points_img1.pop_back();
        }
        else {
          data->points_img2.pop_back();
        }
      }
      else { //save new point
        if (x < data->cols_img1) {
          data->points_img1.push_back(Point(x, y));
        }
        else {
          data->points_img2.push_back(Point(x - data->cols_img1, y));
        }
      }
      drawPoints(userdata);
      break;
    default:
      break;
  }
}


/**
* Default Constructor
*/
ManualRegistration::ManualRegistration() {
  window_name_ = "init";
}


/**
* Constructor
*
* @param img1         First image to register
* @param img2         Second image to register
* @param window_name  Window to display images in
*/
ManualRegistration::ManualRegistration(Mat img1, Mat img2, String window_name) {
  img1_ = img1;
  img2_ = img2;
  window_name_ = window_name;
}


/**
* Set/Reset parameters
*
* @param img1         First image to register
* @param img2         Second image to register
* @param window_name  Window to display images in
*/
void ManualRegistration::setParams(Mat img1, Mat img2, String window_name) {
  img1_ = img1;
  img2_ = img2;
  window_name_ = window_name;
  points_img1_.clear();
  points_img2_.clear();
}


double distToTransPoint(Point& p1, Point& p2, Mat& H) {
  Mat_<double> p(3, 1);
  p(0, 0) = p1.x;
  p(1, 0) = p1.y;
  p(2, 0) = 1;

  Mat_<double> pH = H * p;
  double a, b;
  a = pH(0, 0) / pH(2, 0);
  b = pH(1, 0) / pH(2, 0);

  return sqrt((a - p2.x) * (a - p2.x) + (b - p2.y) * (b - p2.y));
}

double distToEpiline(Point& p1, Point& p2, Mat& F) {
  Mat_<double> p(3, 1);
  p(0, 0) = p1.x;
  p(1, 0) = p1.y;
  p(2, 0) = 1;

  Mat_<double> epiLine = F * p;
  double a, b, c;
  a = epiLine(0, 0);
  b = epiLine(1, 0);
  c = epiLine(2, 0);

  return abs(a * p2.x + b * p2.y + c) / (sqrt(a * a + b * b));
}

/**
* Registers two images via manual corresponding point selection.
*
* @param H   Calculated Homography/Fundamental Matrix from first to second image
* @param cH  true: compute Homography
*            false: compute Fundamental Matrix
* @return    Depicts if computation succeeded
*/
bool ManualRegistration::registerImages(cv::Mat& H, int& nPoints, int& nInliers,
    double& meanDist, bool cH) {
  if (window_name_ == "init") {
    cout << "WARNING: Please first set parameters " <<
        "via setParams(img1, img2, ?window_name)" << endl;
    return false;
  }
  else {
    namedWindow(window_name_, WINDOW_AUTOSIZE);
    displayImagesSideBySide();

    RegistParams params;
    params.cols_img1 = img1_.cols;
    params.points_img1 = points_img1_;
    params.points_img2 = points_img2_;
    params.img_pair = img_pair_;
    params.window_name = window_name_;
    setMouseCallback(window_name_, mouseHandler, (void*)&params);
    cout << "Please select at least ";
    if (cH) cout << "4 ";
    else cout << "8 ";
    cout << "corresponding point pairs. " <<
        "To delete the last point selected press SHIFT and click on " <<
        "the image to delete the point from." << endl <<
        "When done hit any key." << endl;
    waitKey(0);
    points_img1_ = params.points_img1;
    points_img2_ = params.points_img2;

    while (points_img1_.size() > points_img2_.size()) {
      points_img1_.pop_back();
      cout << "WARNING: selected more points in left image" << endl;
    }
    while (points_img1_.size() < points_img2_.size()) {
      points_img2_.pop_back();
      cout << "WARNING: selected more points in right image" << endl;
    }

    if (cH) {
      // calculate Homography
      if (points_img1_.size() >= 4) {
        vector<uchar> inlier_mask;
        H = findHomography(points_img1_, points_img2_, 0, 5, inlier_mask);
        nPoints = (int)points_img1_.size();
        //count inliers
        nInliers = 0;
        double Dist = 0;
        for (size_t i = 0; i < inlier_mask.size(); i++) {
          if (inlier_mask[i] == 1) {
            nInliers++;
            Dist += distToTransPoint(points_img1_[i], points_img2_[i], H);
          }
        }
        meanDist = Dist / nInliers;
        destroyWindow(window_name_);
        return true;
      }
      else {
        cout << "ABORT: Less than 4 point pairs where selected" << endl;
        destroyWindow(window_name_);
        return false;
      }
    }
    else {
      if (points_img1_.size() >= 8) {
        vector<uchar> inlier_mask;
        //param1 = 3 max dist from point to epipolar line to be considered an inlier
        //param2 = 0.99 confidence that the estimated matrix is correct
        H = findFundamentalMat(points_img1_, points_img2_, CV_FM_RANSAC,
            5, 0.99, inlier_mask);
        nPoints = (int)points_img1_.size();
        //count inliers
        nInliers = 0;
        double Dist = 0;
        for (size_t i = 0; i < inlier_mask.size(); i++) {
          if (inlier_mask[i] == 1) {
            nInliers++;
            Dist += distToEpiline(points_img1_[i], points_img2_[i], H);
          }
        }
        meanDist = Dist / nInliers;
        destroyWindow(window_name_);
        return true;
      }
      else {
        cout << "ABORT: Less than 8 point pairs where selected" << endl;
        destroyWindow(window_name_);
        return false;
      }
    }
  }
}
