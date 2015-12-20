#ifndef ESTIMATION_HPP
#define ESTIMATION_HPP

#include <opencv2/opencv.hpp>
#include "ImageSeries.hpp"
#include <vector>

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
 * @return A new vector containing only the valid points from v, in the same * order
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

#endif /* end of include guard: ESTIMATION_HPP */
