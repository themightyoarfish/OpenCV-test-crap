#include "estimation.hpp"
#include "utils.h"
#include <stdexcept>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;
using namespace std;
namespace relative_pose 
{

   const Point2f INVALID_PT = Point2f(-1,-1);

   inline vector<DMatch> ratio_test(const Mat descriptors1, const Mat descriptors2,
         const float ratio, detector_type dtype)
   {
      /* Lambda for sorting matches descending according to distance */
      auto match_comparator = [](DMatch& m1, DMatch& m2) { return m1.distance < m2.distance; };

      vector<vector<DMatch>> candidates;
      BFMatcher matcher;
      switch (dtype)
      {
         case DETECTOR_SIFT:
            matcher = BFMatcher(NORM_L2); // Change for AKAZE
            break;
         case DETECTOR_KAZE:
            matcher = BFMatcher(NORM_HAMMING); // Change for AKAZE
            break;
         default:
            throw runtime_error("Detector not implemented.");
      }
      matcher.knnMatch(descriptors1, descriptors2, candidates, 2); // 2 best matches
      vector<DMatch> matches;
      for (unsigned int i = 0; i < candidates.size(); i++)
      {
         DMatch& m1 = candidates[i][0];
         DMatch& m2 = candidates[i][1];
         if (m1.distance < ratio * m2.distance)
            matches.push_back(m1); // use each match passing the ratio test
      }

      /* Sort matches according to goodness/stability/distance */
      std::sort(matches.begin(), matches.end(), match_comparator);
      return matches;
   }

   void drawMatches(vector<Point2f> pts1, vector<Point2f> pts2, Mat img1, Mat img2)
   {
      /* Fail if vectors have different size */
      if (not (pts1.size() == pts2.size()))
      {
         const size_t bufsz = 256;
         char buffer[bufsz];
         snprintf(buffer, bufsz, "Point vectors are of different length (%lu and %lu)",
               pts1.size(), pts2.size());
         throw invalid_argument(buffer);
      }

      vector<KeyPoint> k1(pts1.size()), k2(pts2.size());
      vector<DMatch> matches(pts1.size());

      for (int i = 0; i < pts1.size(); ++i) matches[i] = DMatch(i,i,0); // points are ordered, so match trivially

      const int KEYPT_SIZE = 4;
      /* points -> keypoints */
      std::transform(pts1.begin(), pts1.end(), k1.begin(), [](Point2f& p) { return KeyPoint(p, KEYPT_SIZE); });
      std::transform(pts2.begin(), pts2.end(), k2.begin(), [](Point2f& p) { return KeyPoint(p, KEYPT_SIZE); });

      /* Draw the matches */
      Mat matchesImg;
      drawMatches(img1,k1,img2,k2,matches,matchesImg);

      /* Show the window */
      const string WIN_NAME = "Matching";
      namedWindow(WIN_NAME, WINDOW_NORMAL);
      imshow(WIN_NAME, matchesImg);
      waitKey(0);
   }

   inline vector<Point2f> remove_invalid(vector<Point2f>& v)
   {
      auto new_end = std::remove_if(v.begin(), v.end(), [](Point2f& p) { return p == INVALID_PT; });
      return vector<Point2f>(v.begin(), new_end);
   }

   /**
    * @brief Convert two sets of keypoints and mathces between them to two vectors
    * of points where corresponding points are at the same index.
    *
    * @param matches The matches indexing into the keypoint arrays
    * @param kpts1 First keypoint set
    * @param kpts2 Second keypoint set
    * @return A tuple with the ordered point vectors
    */
   tuple<vector<Point2f>, vector<Point2f>>
      matches_to_points(vector<DMatch>& matches, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2)
      {
         const int N = matches.size();
         vector<Point2f> imgpts1(N), imgpts2(N);
         for(unsigned int i = 0; i < N; i++) 
         {
            imgpts1[i] = kpts1[matches[i].queryIdx].pt; 
            imgpts2[i] = kpts2[matches[i].trainIdx].pt; 
         }
         return std::make_tuple(imgpts1, imgpts2);
      }

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
   PoseData relative_pose(Mat& descriptors_first, vector<Point2f> pts_first, Mat& _3d_pts, Mat& first_frame, Mat& current_frame, Ptr<Feature2D> detector, float ratio, Mat& camera_matrix, bool show_matches = false)
   {
      Mat rvec, t_first_current; // note the indices are reversed compared to my thesis
      Mat R_first_current;
      Mat descriptors_current;
      vector<KeyPoint> kpts_current;
      detector->detectAndCompute(current_frame, noArray(), kpts_current, descriptors_current); // TODO: make a new detector
      vector<DMatch>  matches_first_current = ratio_test(descriptors_first, descriptors_current, ratio);

      vector<Point2f> pts_current;
      pts_current.resize(kpts_current.size(), INVALID_PT);

      for (DMatch& m : matches_first_current) // convert to points
         pts_current[m.queryIdx] = kpts_current[m.trainIdx].pt;

      Mat _3d_pts_good_subset;               // world points filtered so that each one occurs in current_frame
      vector<Point2f> pts_current_copy;      // img points in current_frame which have match in ff points
      vector<Point2f> pts_first_for_current; // filtered ff points

      for (unsigned int i = 0; i < pts_first.size(); ++i)
      {
         if (not (pts_current[i] == INVALID_PT))
         {
            pts_current_copy.push_back(pts_current[i]);
            _3d_pts_good_subset.push_back(_3d_pts.row(i));
            pts_first_for_current.push_back(pts_first[i]);
         }
      }
      pts_current = pts_current_copy;

#ifdef DEBUG
      std::cout << "pts_first_for_current: " << pts_first_for_current.size() << std::endl;
      std::cout << "_3d_pts_good_subset: " << _3d_pts_good_subset.size() << std::endl;
#endif
      if (show_matches) drawMatches(pts_first_for_current, pts_current, first_frame, current_frame);

      solvePnP(_3d_pts_good_subset, pts_current, camera_matrix, noArray(), rvec, t_first_current);
      Rodrigues(rvec,R_first_current);

#ifdef DEBUG
      std::cout << "Motion first -> current: \n" << PoseData(R_first_current, t_first_current).to_string() << std::endl;
#endif
      return PoseData(R_first_current, t_first_current);
   }

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
   vector<PoseData> runEstimateAuto(const ImageSeries& series, bool show_matches, unsigned int resize_factor, detector_type dtype)
   {
      unsigned int n; // Number of points detected in first frame

      vector<KeyPoint> kpts_first, kpts_second, kpts_ref; // keypoint vectors
      vector<Point2f>  pts_first,                         // all ff kpts converted to points, later filtered with second frame
         pts_second, pts_ref;                             // all 2nd and ref frame kpts, later filtered so each has matches in ff
      Mat descriptors_first, descriptors_second, descriptors_ref;
      Ptr<Feature2D> detector;
      switch (dtype)
      {
         case DETECTOR_SIFT:
            std::cout << "Detector: SIFT" << std::endl;
            detector = xfeatures2d::SIFT::create(); // SIFT or AKAZE
            break;
         case DETECTOR_KAZE:
            std::cout << "Detector: AKAZE" << std::endl;
            detector = AKAZE::create(); // SIFT or AKAZE
            break;
         case DETECTOR_SURF:
            std::cout << "Detector: SURF" << std::endl;
            detector = xfeatures2d::SURF::create(); // SIFT or AKAZE
            break;
         default:
            throw runtime_error("Detector not currently supported.");
      }

      Mat first_frame, second_frame, reference_frame;

      /* Parameter for ratio test */
      const float RATIO = 0.8;

      BFMatcher matcher;

      /* Scale down images */
      if (resize_factor > 1)
      {
         resize(series.first_frame(),     first_frame,     Size(), 1. / resize_factor,  1. / resize_factor);
         resize(series.second_frame(),    second_frame,    Size(), 1. / resize_factor,  1. / resize_factor);
         resize(series.reference_frame(), reference_frame, Size(), 1. / resize_factor,  1. / resize_factor);
      } else 
      {
         first_frame     = series.first_frame();
         second_frame    = series.second_frame();
         reference_frame = series.reference_frame();
      }

      detector->detectAndCompute(first_frame,  noArray(), kpts_first,  descriptors_first);
      detector->detectAndCompute(second_frame, noArray(), kpts_second, descriptors_second);

      n = kpts_first.size();
      /* Since we always compare w/ first frame, make all same size, fill with
       * invalidity markers
       */
      pts_first.resize(n,   INVALID_PT);
      pts_second.resize(n,  INVALID_PT);
      pts_ref.resize(n,     INVALID_PT);

      /* Convert keypoints to points */
      std::transform(kpts_first.begin(), kpts_first.end(), pts_first.begin(), [&](KeyPoint& k) { return k.pt; });

      /* Ratio test the first and second frame */
      vector<DMatch> matches_first_second = ratio_test(descriptors_first, descriptors_second, RATIO);

      for (DMatch& m : matches_first_second)
         pts_second[m.queryIdx] = kpts_second[m.trainIdx].pt; // order the points so their match is at the same index

      /********* FILTER FIRST FRAME WITH SECOND FRAME *******************************
        Filter the first frame's descriptors, points and keypoints
        based on existence of correspondence in second frame
      ******************************************************************************/
      Mat descriptors_first_filtered;
      vector<Point2f> pts_first_copy, pts_second_copy;
      vector<KeyPoint> kpts_first_copy;
      for (unsigned int i = 0; i < kpts_first.size(); ++i)
      {
         if (not (pts_second[i] == INVALID_PT)) // copy each index where both have valid point
         {
            pts_second_copy.push_back(pts_second[i]);
            pts_first_copy.push_back(pts_first[i]);
            kpts_first_copy.push_back(kpts_first[i]); // also filter keypoints
            descriptors_first_filtered.push_back(descriptors_first.row(i)); // push the corresponding descriptor
         }
      }
      /* Move the copies into the originals */
      pts_first  = pts_first_copy;
      kpts_first = kpts_first_copy;
      pts_second = pts_second_copy;
      descriptors_first.release();
      descriptors_first_filtered.copyTo(descriptors_first);
      /********* FILTERING DONE ****************************************************/

      Mat_<double> camera_matrix    = series.camera_matrix() / resize_factor;
      camera_matrix(2,2) = 1;
      Mat dist_coeffs               = series.dist_coeffs();
      double focal                  = camera_matrix(0,0);
      Point2d principalPoint(camera_matrix(0,2), camera_matrix(1,2));

      Mat mask;
      Mat E = findEssentialMat(pts_first, pts_second, 
            focal, principalPoint,
            RANSAC,
            0.99, // confidence
            3, // distance to be considered outlier
            mask);

      Mat R_first_second, t_first_second;
      int inliers = recoverPose(E, pts_first, pts_second, R_first_second, t_first_second, focal, principalPoint, mask);

#ifdef DEBUG
      std::cout << "Motion first -> second: \n" << PoseData(R_first_second, t_first_second).to_string() << std::endl;
#endif

      /********* FILTER FIRST AND SECOND FRAME WITH MASK ***************************
        Filter the first frame's descriptors, points and keypoints
        and the second frame's points with the mask from findessentialmat
       *****************************************************************************/
      /* clear old copies */
      descriptors_first_filtered.release();  // probably redundant
      descriptors_first_filtered = Mat();
      pts_first_copy.clear();
      pts_second_copy.clear();
      kpts_first_copy.clear();
      for (unsigned int i = 0; i < mask.rows; ++i)
      {
         if (mask.at<uchar>(i,0) == 1) // if point is inlier
         {
            kpts_first_copy.push_back(kpts_first[i]);
            pts_first_copy.push_back(pts_first[i]);
            pts_second_copy.push_back(pts_second[i]);
            descriptors_first_filtered.push_back(descriptors_first.row(i));
         }
      }
      pts_first  = pts_first_copy;
      kpts_first = kpts_first_copy;
      pts_second = pts_second_copy;
      descriptors_first.release();
      descriptors_first_filtered.copyTo(descriptors_first);
      /********* FILTERING DONE ****************************************************/

      if (show_matches) drawMatches(pts_first, pts_second, first_frame, second_frame);

      /********* 3D POINT TRIANGULATION ********************************************/
      Mat pnts4D;
      Mat P1    = camera_matrix * Mat::eye(3, 4, CV_64FC1); // first projection matrix
      Mat P2;                                               // second projection matrix
      Mat p2[2] = { R_first_second, t_first_second };       // concat R and t to make P2
      hconcat(p2, 2, P2);
      P2 = camera_matrix * P2;                              // remove camera imperfections
      triangulatePoints(P1, P2, pts_first, pts_second, pnts4D);
      pnts4D = pnts4D.t();
      Mat _3d_points;
      convertPointsFromHomogeneous(pnts4D, _3d_points);
      _3d_points = _3d_points.reshape(1);             // instead of 3 channels and 1 col, we want 1 channel and 3 cols

      /*** SOLVEPNP ***/
#ifdef DEBUG
      std::cout << "First -> Reference " << std::endl;
#endif
      PoseData d = relative_pose(descriptors_first, pts_first, _3d_points, 
            first_frame, reference_frame, detector, RATIO, camera_matrix, show_matches);
      Mat t_first_ref = d.t, 
          R_first_ref = d.R;

      const unsigned int numImgs = series.num_intermediate_imgs();
      vector<PoseData> ret(numImgs);

      for (unsigned int i = 0; i < numImgs; ++i) 
      {

         Mat current_frame;
         if (resize_factor > 1)
            resize(series.image_for_index(i), current_frame, Size(), 1. / resize_factor,  1. / resize_factor);
         else current_frame = series.image_for_index(i);

#ifdef DEBUG
         std::cout << "First -> Frame " << i << std::endl;
#endif
         PoseData d = relative_pose(descriptors_first, pts_first, _3d_points, 
               first_frame, current_frame, detector, RATIO, camera_matrix, show_matches);
         Mat t_first_current = d.t,
             R_first_current = d.R;
         Mat t_current_ref   = -R_first_ref * R_first_current.t() * t_first_current + t_first_ref;
         Mat R_current_ref   = R_first_ref * R_first_current.t();
         ret[i] = PoseData(R_current_ref, t_current_ref);
      }

      return ret;
   }

   void convertToKeypoints(const CorrVec& v, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, 
         float size, float angle, float response, int octave, int classid)
   {
      unsigned int i = 0;
      kpts1.resize(v.size());
      kpts2.resize(v.size());
      for (auto iter = v.begin(); iter != v.end(); iter++, i++) 
      {
         const pair<Point2i,Point2i>& corresp_pts = *iter;
         Point2f&& p1 = Point2f(corresp_pts.first.x,corresp_pts.first.y);
         Point2f&& p2 = Point2f(corresp_pts.second.x,corresp_pts.second.y);
         kpts1[i] = KeyPoint(p1, size, angle, response, octave, classid);
         kpts2[i] = KeyPoint(p2, size, angle, response, octave, classid);
      }
   }
}

