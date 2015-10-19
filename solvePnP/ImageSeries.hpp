#ifndef IMAGE_SERIES_H
#define IMAGE_SERIES_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImageSeries
{

   private:
      std::vector<cv::Mat> images;

   public:
   ImageSeries(cv::Mat&& first_frame, cv::Mat&& second_frame, cv::Mat&& reference);
   ImageSeries(std::vector<cv::Mat> mats);
   cv::Mat& first_frame(void);
   cv::Mat& second_frame(void);
   cv::Mat& reference_frame(void);
   const std::vector<cv::Mat>& frames(void);
   void set_images(std::vector<cv::Mat> imgs);
};
#endif
