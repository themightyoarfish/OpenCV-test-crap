/*
 * author: Ann-Katrin Häuser <ahaeuser@uos.de>
 *
 * Application to manually register either two given images
 * or all images in a given directory.
 * If a directory is passed to the application such is searched recursively
 * for images without matching information and a manual matching routine
 * is called for such. The results are saved to a yaml file which is named
 * in accordance with the image pair and placed in the same folder.
 */

#include "manual_registration/slider.h"
#include "manual_registration/manual_registration.h"
#include "image_data_handling/pair_finder.h"
#include "image_data_handling/yml_read_write.h"

#include <iostream>
#include <boost/algorithm/string.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;


// folder to search for images by default
string default_data = "../../rephoto_data/images";

enum {
  F_NAME_YEAR = 0,
  F_NUMBER = 1,
};

/**
* Registers two images via manual corresponding point selection.
* Show the result in a Slider Window
*
* @param H   calculated Homography between both images
* @param cH  true: compute Homography
*            false: compute Fundamental Matrix
*/
void register_images_manually(Mat img1, Mat img2, Mat& H,
    int& nPoints, int& nInliers, double& meanDist, bool cH = true,
    bool iter = false, string storage = "") {
  Mat img1_resized;
  Mat img2_resized;
  double scale = 1.0;
  if (max(img1.cols, img1.rows) > 1500 && cH) {
    scale = 1500.0 / max(img1.cols, img1.rows);
    resize(img1, img1_resized, Size(), scale, scale);
    resize(img2, img2_resized, Size(), scale, scale);
    img1 = img1_resized;
    img2 = img2_resized;
  }

  ManualRegistration mreg(img1, img2, "ImagePair");
  Slider slider;
  bool done = false;
  do {
    if (mreg.registerImages(H, nPoints, nInliers, meanDist, cH)) {
      cout << "selectedPoints: " << nPoints << " Inliers: " << nInliers << endl
          << "MeanDistance: " << meanDist << endl;
      if (cH) {
        // apply Homography and diplay with Slider
        Mat warp_img;
        warpPerspective(img1, warp_img, H, img1.size());
        slider.setParams(warp_img, img2);
      }
      else {
        // rectify both images to determine sufficiency of fundamental matrix
        Mat H1, H2, rec_img1, rec_img2;
        stereoRectifyUncalibrated(mreg.getPointsImg1(), mreg.getPointsImg2(),
            H, img1.size(), H1, H2);
        warpPerspective(img1, rec_img1, H1, img1.size());
        warpPerspective(img2, rec_img2, H2, img2.size());
        slider.setParams(rec_img1, rec_img2);
      }
      slider.displaySliderWindow();
      string ans;
      cout << "Is the calculated Homography/Fundamentalmatrix sufficient? (y,n) ";
      cin >> ans;
      if (ans == "y") {
        done = true;
      }
    }
  }
  while (!done);
  if (scale != 1.0 && cH) {
    Mat Sdown = Mat::zeros(3, 3, H.type());
    Mat Sup = Mat::zeros(3, 3, H.type());
    Sdown.at<double>(0, 0) = Sdown.at<double>(1, 1) =
        scale;
    Sdown.at<double>(2, 2) = 1;
    float lamda = float(1.0 / scale);
    Sup.at<double>(0, 0) = Sup.at<double>(1, 1) =
        lamda;
    Sup.at<double>(2, 2) = 1;
    H = Sup * H * Sdown;
  }
}

/**
* Calls manual registration for all given image pairs.
* Results are saved to a yaml file.
*
* @param pairs   Vector of image pairs to register
* @param format  Naming format of the image pairs (number vs name-year)
* @param cH      true: compute Homography
*                false: compute Fundamental Matrix
*/
void register_all_pairs(vector<rephoto::ImgPairInfo>& pairs, int format,
    bool cH = true) {
  Mat H, img1, img2;
  path p1, p2;
  int year1, year2;
  vector<string> segments;
  string store;
  string category;
  int nPoints, nInliers;
  double meanDist;
  for (int i = 0; i < pairs.size(); i++) {
    img1 = imread(pairs[i].path_img1, 1);
    img2 = imread(pairs[i].path_img2, 1);
    if (!img1.data || !img2.data) {
      cout << " --(!) Unable to read at least one of these images: "
          << endl << pairs[i].path_img1 << endl << pairs[i].path_img2
          << endl;
    }
    else {
      cout << "current pair: " << endl << pairs[i].path_img1 << endl
          << pairs[i].path_img2 << endl;
      register_images_manually(img1, img2, H, nPoints, nInliers, meanDist, cH);
      p1 = path(pairs[i].path_img1);
      p2 = path(pairs[i].path_img2);
      switch (format) {
        case F_NUMBER:
          store = string(p2.parent_path().c_str()) + "/H-" +
              p1.stem().c_str() + "-" + p2.stem().c_str() + ".yml";
          category = rephoto::selectCategory();
          cout << "save to: " << store << endl;
          rephoto::saveRegistInfo2yaml(H, p1.filename().c_str(),
              p2.filename().c_str(), nPoints, nInliers, meanDist, cH,
              store, category);
          break;
        case F_NAME_YEAR:
          boost::split(segments, p1.stem().string(), boost::is_any_of("-"));
          year1 = atoi(segments[1].c_str());
          boost::split(segments, p2.stem().string(), boost::is_any_of("-"));
          year2 = atoi(segments[1].c_str());
          store = string(p2.parent_path().c_str()) + "/"
              + segments[0] + ".yml";
          cout << "save to: " << store << endl;
          category = rephoto::selectCategory();
          rephoto::saveRegistInfo2yaml(H, p1.filename().c_str(),
              p2.filename().c_str(), nPoints, nInliers, meanDist, cH, store,
              category, year1, year2);
          break;
      }
    }
  }
}

/**
* Calls manual registration for all given image sets.
* Results are saved to a yaml file.
*
* @param set     Vector of image sets to register
* @param cH      true: compute Homography
*                false: compute Fundamental Matrix
*/
void register_all_sets(vector<rephoto::ImgSetInfo>& sets, bool cH = true) {
  Mat H, img_org, img2;
  path p1, p2;
  string store;
  string category;
  int nPoints, nInliers;
  double meanDist;
  for (size_t i = 0; i < sets.size(); i++) {
    img_org = imread(sets[i].path_org, 1);
    if (!img_org.data) {
      cout << " --(!) Unable to read " << sets[i].path_org << endl;
    }
    else {
      for (size_t j = 0; j < sets[i].path_imglist.size(); j++) {
        img2 = imread(sets[i].path_imglist[j], 1);
        if (!img2.data) {
          cout << " --(!) Unable to read " << sets[i].path_imglist[j] << endl;
        }
        else {
          register_images_manually(img_org, img2, H, nPoints, nInliers,
              meanDist, cH);
          p1 = path(sets[i].path_org);
          p2 = path(sets[i].path_imglist[j]);
          store = string(p2.parent_path().c_str()) + "/H-" +
              p1.stem().c_str() + "-" + p2.stem().c_str() + ".yml";
          category = rephoto::selectCategory();
          rephoto::saveRegistInfo2yaml(H, p1.filename().c_str(),
              p2.filename().c_str(), nPoints, nInliers, meanDist, cH,
              store, category);
        }
      }
    }
  }
}


/**
* Usage Instructions
*/
void readme() {
  cout << " Usage: ./ManualPointRegistration <img1> <img2>" << endl;
  cout << "        ./ManualPointRegistration <img1> <img2> <destination.yml>"
      << endl;
  cout << "        ./ManualPointRegistration <img1> <img2> <destination.yml>"
      << " <warp_img1>" << endl;
  cout << "        ./ManualPointRegistration "
      << "(searches for image pairs in complete data folder)" << endl;
}

int main(int argc, char** argv) {
  Mat H;
  Mat img1, img2;
  vector<rephoto::ImgPairInfo> pairs;
  vector<rephoto::ImgSetInfo> sets;
  string directory;
  switch (argc - 1) {
    //register two given images
    case 2:
    case 3:
    case 4:
      img1 = imread(argv[1], 1);
      img2 = imread(argv[2], 1);
      if (!img1.data || !img2.data) {
        cout << " --(!) Unable to read images " << endl;
        return -1;
      }
      int nPoints, nInliers;
      double meanDist;
      register_images_manually(img1, img2, H, nPoints, nInliers, meanDist);
      if (argc < 4) {
        rephoto::saveRegistInfo2yaml(H, argv[1], argv[2], nPoints, nInliers,
            meanDist);
      }
      else {
        rephoto::saveRegistInfo2yaml(H, argv[1], argv[2], nPoints, nInliers,
            meanDist, true, argv[3]);
      }
      if (argc == 4 + 1) {
        Mat warp_img;
        img1 = imread(argv[1], 1);
        warpPerspective(img1, warp_img, H, img1.size());

        vector<Point2f> imgCorners(4);
        imgCorners[0] = cvPoint(0, 0);
        imgCorners[1] = cvPoint(img1.cols, 0);
        imgCorners[2] = cvPoint(img1.cols, img1.rows);
        imgCorners[3] = cvPoint(0, img1.rows);
        vector<Point2f> proj1(4);
        perspectiveTransform(imgCorners, proj1, H);


        int x = max(proj1[0].x, proj1[3].x);
        x = max(x, 0);
        int y = max(proj1[0].y, proj1[1].y);
        y = max(y, 0);
        int width = min(proj1[1].x, proj1[2].x);
        width = min(width, img1.size().width) - x;
        int height = min(proj1[2].y, proj1[3].y);
        height = min(height, img1.size().height) - y;
        std::cout << x << "," << y << "," << width << "," << height << std::endl;
        cv::Mat warpimgcut(warp_img, Rect(x, y, width, height));
        cv::Mat img2cut(img2, Rect(x, y, width, height));

        imwrite(argv[4], warp_img);
        imwrite("img1_warp_cut.jpg", warpimgcut);
        imwrite("img2_cute.jpg", img2cut);
      }
      break;
      // search and register image pairs in complete data folder
    case 0:
      directory = default_data;
      directory += "/vornberger";
      rephoto::PairFinder::search4ImagePairs(directory, rephoto::MISSING_YAML,
                                             pairs);
      cout << "Found " << pairs.size() << " image pairs without yml file "
      << "in dataset vornberger" << endl
      << "Start manual registration!" << endl;
      register_all_pairs(pairs, F_NAME_YEAR);

      /*
      directory = default_data;
      directory += "/nyc_grid";
      rephoto::PairFinder::search4ImagePairs(directory, rephoto::MISSING_YAML,
          pairs);
      cout << "Found " << pairs.size() << " image pairs without yml file "
          << "in dataset nyc_grid" << endl
          << "Start manual registration!" << endl;
      register_all_pairs(pairs, F_NAME_YEAR);

      directory = default_data;
      directory += "/symbench_v1";
      rephoto::PairFinder::search4ImagePairs(directory, rephoto::MISSING_YAML,
          pairs);
      cout << "Found " << pairs.size() << " image pairs without yml file "
          << "in dataset symbench_v1" << endl
          << "Start manual registration!" << endl;
      register_all_pairs(pairs, F_NUMBER);

      directory = default_data;
      directory += "/osna_postkarten";
      rephoto::PairFinder::search4ImageSets(directory, rephoto::MISSING_YAML,
          sets);
      cout << "Found " << sets.size() << " image sets with missing yml files "
          << "in dataset osna_postkarten" << endl
          << "Start manual registration!" << endl;
      register_all_sets(sets, false);
       */
      break;

    default:
      readme();
  }
  return 0;
}


