#ifndef IMAGE_SERIES_H
#define IMAGE_SERIES_H

#include <opencv2/opencv.hpp>
#include <vector>

typedef std::vector<std::pair<cv::Point2i,cv::Point2i>> CorrVec;

static double cam_mat_data[] = {1.,0.,0.,0.,1.,0.,0.,0.,1.};
static cv::Mat default_cam_mat(3,3, CV_64FC1, cam_mat_data);
void convertToKeypoints(CorrVec& v, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, 
      float size = 10, float angle = -1, float response = 0, int octave = 0, int classid = -1);

class ImageSeries
{
   public:
      enum ImageRole 
      {
         FIRST_FRAME  = 0,
         SECOND_FRAME = 1,
         REF_FRAME    = 2
      };

   private:
      std::vector<cv::Mat> mImages;
      std::vector<CorrVec> mCorrespondences;
      cv::Mat mCameraMatrix;
      cv::Mat mDistCoeffs;
      static void sortPoints(CorrVec& v);

   public:
      ImageSeries(cv::Mat&& first_frame, cv::Mat&& second_frame, cv::Mat&& reference, 
            cv::Mat&& camera_matrix = std::move(default_cam_mat), cv::Mat&& dist_coeffs=cv::Mat(1,5,0));
      ImageSeries(std::vector<cv::Mat> mats);
      cv::Mat& first_frame(void);
      cv::Mat& second_frame(void);
      cv::Mat& reference_frame(void);
      cv::Mat& image_for_index(unsigned int index);
      const std::vector<cv::Mat> frames(void);
      void add_image(cv::Mat img);
      void add_image(cv::Mat img, CorrVec v);
      void add_correspondences(ImageRole role, CorrVec v);
      void add_correspondences(unsigned int index, CorrVec v);
      CorrVec& correspondences_for_frame(ImageRole role);
      CorrVec& correspondences_for_frame(unsigned int index);
      void set_images(std::vector<cv::Mat> imgs);
      cv::Mat& camera_matrix();
      cv::Mat& dist_coeffs();
      /** TODO: MAKE PRIVATE **/
      void showMatches(unsigned int indexl, unsigned int indexr, CorrVec& v);
};
#endif
