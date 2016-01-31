#ifndef IMAGE_SERIES_H
#define IMAGE_SERIES_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"
namespace relative_pose 
{

class ImageSeries;

typedef std::vector<std::pair<cv::Point2f,cv::Point2f>> CorrVec;

/* Eye matrix */
static double cam_mat_data[] = {1.,0.,0.,0.,1.,0.,0.,0.,1.};
static const cv::Mat DEFAULT_CAM_MAT(3,3, CV_64FC1, cam_mat_data);

void convertToKeypoints(const CorrVec& v, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, 
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
      void insertAndUndistort(cv::Mat m, unsigned int index);
      void insertAndUndistort(cv::Mat m);
      void undistortCorrespodences(CorrVec& v);

   public:
      explicit ImageSeries(const cv::Mat&& first_frame, const cv::Mat&& second_frame, const cv::Mat&& reference_frame, 
            const cv::Mat&& camera_matrix = std::move(DEFAULT_CAM_MAT), const cv::Mat&& dist_coeffs=cv::Mat(1,5,0));
      explicit ImageSeries(std::vector<cv::Mat> mats);
      const cv::Mat& first_frame(void)                   const;
      const cv::Mat& second_frame(void)                  const;
      const cv::Mat& reference_frame(void)               const;
      const cv::Mat& image_for_index(unsigned int index) const;
      const std::vector<cv::Mat> frames(void)            const;
      void add_image(cv::Mat img);
      void add_image(cv::Mat img, CorrVec v);
      void add_correspondences(ImageRole role, CorrVec v);
      void add_correspondences(unsigned int index, CorrVec v);
      void set_images(std::vector<cv::Mat> imgs);
      const CorrVec& correspondences_for_frame(ImageRole role)     const;
      const CorrVec& correspondences_for_frame(unsigned int index) const;
      const cv::Mat& camera_matrix(void)                           const;
      const cv::Mat& dist_coeffs(void)                             const;
      unsigned int num_intermediate_imgs(void) const;
      /** TODO: MAKE PRIVATE **/
      void show_matches(unsigned int indexl, unsigned int indexr, const CorrVec& v) const;
};
}
#endif
