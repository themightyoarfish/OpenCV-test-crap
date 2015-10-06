/*
 * author: Ann-Katrin Häuser <ahaeuser@uos.de>
 */

#ifndef REPHOTO_MANUAL_REGISTRATION_H_
#define REPHOTO_MANUAL_REGISTRATION_H_

#include <opencv2/core.hpp>

/**
* Allows registration of two images via manual corresponding point selection.
* Takes two images of the same scene and displays them side by side.
* Now the user is able to mark corresponding points on the basis of which
* both images are registered against each other.
*/
class ManualRegistration {
private:
  // struct to pass paramters to static mouseHandler
  struct RegistParams {
    int cols_img1;
    std::vector<cv::Point> points_img1;
    std::vector<cv::Point> points_img2;
    cv::Mat img_pair;
    cv::String window_name;
    // vector of colors for drawing corresponding points
    const cv::Scalar color[8] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 128, 255),
        cv::Scalar(0, 210, 210), cv::Scalar(0, 210, 110),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 0),
        cv::Scalar(255, 0, 130), cv::Scalar(255, 0, 255)};
  };
  cv::Mat img1_, img2_; // images to be registered
  cv::Mat img_pair_;   // both_images next to each other
  // vectors for saving the selected corresponding points
  std::vector<cv::Point> points_img1_;
  std::vector<cv::Point> points_img2_;
  // window to display images in
  cv::String window_name_;

  /**
  * Displays image pair next to each other and saves it in image_pair
  */
  void displayImagesSideBySide();

  /**
  * Draws all previously selected points on top of image_pair
  */
  static void drawPoints(void* userdata);

  /**
  * Handles LeftMouseButtonClicks to mark corresponding points images.
  * The last selection is deleted upon additional press of the SHIFTKEY
  */
  static void mouseHandler(int event, int x, int y, int flags, void* userdata);

public:
  /**
  * Default Constructor
  */
  ManualRegistration();

  /**
  * Constructor
  *
  * @param img1         First image to register
  * @param img2         Second image to register
  * @param window_name  Window to display images in
  */
  ManualRegistration(cv::Mat img1, cv::Mat img2,
      cv::String window_name = "ImagePair");

  /**
  * Set/Reset parameters
  *
  * @param img1         First image to register
  * @param img2         Second image to register
  * @param window_name  Window to display images in
  */
  void setParams(cv::Mat img1, cv::Mat img2,
      cv::String window_name = "ImagePair");

  /**
  * Registers two images via manual corresponding point selection.
  *
  * @param H         Calculated Homography/Fundamental Matrix from first to
  *                  second image
  * @param nPoints   number of points selected for computation
  * @param nInliers  number of inliers
  * @param meanDist  mean distance to projected point/epipolar line for all
  *                  inliers
  * @param cH        true: compute Homography
  *                  false: compute Fundamental Matrix
  * @return          Depicts if computation succeeded
  */
  bool registerImages(cv::Mat& H, int& nPoints, int& nInliers,
      double& meanDist, bool cH = true);

  std::vector<cv::Point> getPointsImg1() {
    return points_img1_;
  }

  std::vector<cv::Point> getPointsImg2() {
    return points_img2_;
  }
};

#endif  // REPHOTO_MANUAL_REGISTRATION_H_
