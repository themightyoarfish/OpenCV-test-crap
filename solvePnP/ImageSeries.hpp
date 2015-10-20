#ifndef IMAGE_SERIES_H
#define IMAGE_SERIES_H

#include <opencv2/opencv.hpp>
#include <vector>

typedef std::vector<std::pair<cv::Point2i,cv::Point2i>> CorrVec;

class ImageSeries
{
   enum ImageRole 
   {
      FIRST_FRAME  = 0,
      SECOND_FRAME = 1,
      REF_FRAME    = 2
   };

   private:
      std::vector<cv::Mat> images;
      std::vector<CorrVec> correspondences;

   public:
      ImageSeries(cv::Mat&& first_frame, cv::Mat&& second_frame, cv::Mat&& reference);
      ImageSeries(std::vector<cv::Mat> mats);
      cv::Mat& first_frame(void);
      cv::Mat& second_frame(void);
      cv::Mat& reference_frame(void);
      const std::vector<cv::Mat> frames(void);
      void add_image(cv::Mat img);
      void add_image(cv::Mat img, CorrVec v);
      void add_correspondences(ImageRole role, CorrVec v);
      void add_correspondences(int index, CorrVec v);
      void set_images(std::vector<cv::Mat> imgs);
};
#endif
