#ifndef ESTIMATION_HPP
#define ESTIMATION_HPP

#include <opencv2/opencv.hpp>
#include "ImageSeries.hpp"
#include <vector>
namespace relative_pose 
{

   /**
    * @brief Perform a ratio test matching on two descriptor sets.
    *
    * The descriptors are matched with the L2 norm in case \p detector_type is SIFT
    * or Hamming distance for AKAZE. A match (d1, d2) is accepted if the distance
    * between d1 and d2 is at most \p ratio times the distance between d1 and its
    * second-best match.
    *
    * @param descriptors1 First descriptor set
    * @param descriptors2 Second descriptor set
    * @param ratio The ratio to use
    * @param dtype The type of detector used in obtaining the descriptors so that
    * they can be matched correctly
    * @return A vector of good matches betwee \p descriptors1 and \p descriptors2
    */
   std::vector<cv::DMatch> ratio_test(const cv::Mat descriptors1, const cv::Mat descriptors2,
         const float ratio = 0.8, detector_type dtype=DETECTOR_SIFT);

   /**
    * @brief Draw matches between two images
    * 
    * This function accepts two vectors of points which are assumed to be ordered
    * and of same length. This means that pts1[i] corresponds to pts2[i].
    *
    * @param pts1 First image's points
    * @param pts2 Second image's points
    * @param img1 First image
    * @param img2 Second image
    */
   void drawMatches(std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, cv::Mat img1, cv::Mat img2);

   /**
    * @brief Filter out invalid points from a vector.
    *
    * This function is for convenience to clean up calling code.
    *
    * @param v The vector to strip
    * @return A new vector containing only the valid points from v, in the same order
    */
   inline std::vector<cv::Point2f> remove_invalid(std::vector<cv::Point2f>& v);

   /**
    * @brief Convert keypoint vectors and matches to ordered vectors of Points
    *
    * This function returns vectors of points where corresponding elements in the
    * two vectors are at the same index
    *
    * @param matches Match vector idexing the two keypoint vectors
    * @param kpts1 First image's keypoints
    * @param kpts2 Second image's keypoints
    * @return Tuple of vectors with corresponding points ordered according to \p
    * matches
    */
   std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
      matches_to_points(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2);

   /**
    * @brief TODO
    */
   std::vector<PoseData> runEstimateManual(const ImageSeries& series, bool show_matches=false, unsigned int resize_factor=1);

   /**
    * @brief Compute necessary translation and rotation for each image in an image
    * series. Automatic feature detection is used.
    *
    * This function performs the estimate of how the photographer has to move from
    * the vantage point of each image in an image series. 
    *
    * @param series The ImageSeries to perform the analysis on
    * @param show_matches Flag to indicate whether the matches between each
    * processed image pair should be displayed for inspection
    * @param resize_factor Scale factor to apply to the images to speed up feature
    * detection
    */
   std::vector<PoseData> runEstimateAuto(const ImageSeries& series, bool show_matches=false, unsigned int resize_factor=1, detector_type dtype=DETECTOR_SIFT);

   /**
    * @brief Convert two sets of keypoints and mathces between them to two vectors
    * of points where corresponding points are at the same index.
    *
    * @param matches The matches indexing into the keypoint arrays
    * @param kpts1 First keypoint set
    * @param kpts2 Second keypoint set
    * @return A tuple with the ordered point vectors
    */
   std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
      matches_to_points(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2);

   /**
    * @brief Compute relative pose between two images given as feature descriptors
    * for a reference image and a cv::Mat for the train image.
    *
    * This function computes descriptors from the train image \p current_frame and
    * matches them with the training descriptors (intended to be the firs frame's).
    * It implements the estimation procedure based on finding correspondences
    * between precomputed world points and features found in the \p current_frame
    * and applying cv::solvePnP. It optionally shows the matches finally used for user
    * inspection.
    *
    * @param descriptors1 The reference descriptors 
    * @param pts_first Vector of cv::Point2f of the points belonging to the
    * descriptors.
    * @param _3d_pts The corresponding world points 
    * @param first_frame The image used to obtain the \p descriptors1. Used to
    * display the matches.
    * @param current_frame The image to compute the relative pose for 
    * @param detector The feature detector to use for finding points of interest
    * @param ratio The ratio used for the relative_pose::ratio_test
    * @param camera_matrix The camear intrinsics
    * @param show_matches Whether or not to display matches. Defaults to \c false
    * @return A relative_pose::PoseData object
    */
   PoseData relative_pose(cv::Mat& descriptors_first, std::vector<cv::Point2f> pts_first, cv::Mat& _3d_pts, cv::Mat& first_frame, cv::Mat& current_frame, cv::Ptr<cv::Feature2D> detector, float ratio, cv::Mat& camera_matrix, bool show_matches = false);

   /**
    * @brief Run the estimatino with automatic features 
    *
    * @param series The images to operate on 
    * @param show_matches Whether or not to display the matches between each
    * pair of images 
    * @param resize_factor Scale actor to apply to the images (smaller is much
    * faster, but fewer features and possibly lower quality estimates)
    * @param dtype The type of detector to use 
    * @return An std::vector of relative_pose::PoseData objects, one for each
    * intermediate image
    */
   std::vector<PoseData> runEstimateAuto(const ImageSeries& series, bool show_matches, unsigned int resize_factor, detector_type dtype);
}
#endif /* end of include guard: ESTIMATION_HPP */
