#include <stdexcept>
#include "ImageSeries.hpp"

using namespace std;
using namespace cv;

vector<PoseData> runEstimate(const ImageSeries& series, bool interactive)
{
   const CorrVec& corr_first_second = series.correspondences_for_frame(ImageSeries::SECOND_FRAME);
   const CorrVec& corr_first_ref = series.correspondences_for_frame(ImageSeries::REF_FRAME);
   const unsigned int n = corr_first_second.size();
   vector<KeyPoint> kpts_first(n), kpts_second(n), kpts_ref(n);
   vector<Point2f> pts_first(n), pts_second(n), pts_ref(n);
   convertToKeypoints(corr_first_second, kpts_first, kpts_second);
   convertToKeypoints(corr_first_ref, kpts_first, kpts_ref);
   for (unsigned int i = 0; i < n; ++i) 
   {
      pts_first[i] = kpts_first[i].pt;
      pts_second[i] = kpts_second[i].pt;
      pts_ref[i] = kpts_ref[i].pt;
   }
   Mat camera_matrix = series.camera_matrix();
   Mat dist_coeffs = series.dist_coeffs();

   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat mask;
   Mat E = findEssentialMat(pts_first, pts_second, focal, principalPoint, RANSAC, 0.999, 1, mask);

   Mat R, t;
   int inliers = recoverPose(E, pts_first, pts_second, R, t, focal, principalPoint, mask);
   Vec3d angles = rotationMatToEuler(R);
   
   vector<Point2f> pts_first_masked, pts_second_masked, pts_ref_masked;
   for (int i = 0; i < pts_first.size(); i++) 
   {
      if (mask.at<uchar>(i,0) == 1) 
      {
         pts_first_masked.push_back(pts_first[i]);
         pts_second_masked.push_back(pts_second[i]);
         pts_ref_masked.push_back(pts_ref[i]);
      }
   }

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;
   triangulatePoints(P1, P2, pts_first_masked, pts_second_masked, pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols

   /*** SOLVEPNP ***/
   Mat rvec, t_first_ref; // note the indices are reversed compared to my thesis
   Mat R_first_ref;
   solvePnP(dehomogenized, pts_ref_masked, camera_matrix, noArray(), rvec, t_first_ref);
   Rodrigues(rvec,R_first_ref);

   std::cout << "R_first_ref: " << rotationMatToEuler(R_first_ref) << std::endl;
   std::cout << "t_first_ref: " << t_first_ref << std::endl;

   const unsigned int numImgs = series.num_intermediate_imgs();
   vector<PoseData> ret(numImgs);

   for (unsigned int i = 0; i < numImgs; ++i) 
   {
      CorrVec corr_first_current = series.correspondences_for_frame(i);
      vector<Point2f> pts_current_masked;
      vector<KeyPoint> kpts_current(n);
      convertToKeypoints(corr_first_current, kpts_first, kpts_current);
      for (unsigned int i = 0; i < n; ++i) 
         if (mask.at<uchar>(i,0) == 1) 
            pts_current_masked.push_back(kpts_current[i].pt);
      Mat R_first_current, t_first_current;
      solvePnP(dehomogenized, pts_current_masked, camera_matrix, noArray(), rvec, t_first_current);
      Rodrigues(rvec,R_first_current);
      if (interactive)
      {
         series.show_matches(0, i + 3, series.correspondences_for_frame(i));
      }
      Mat t_current_ref = -R_first_ref * R_first_current.t() * t_first_current + t_first_ref;
      Mat R_current_ref = R_first_ref * R_first_current.t();
      ret[i] = { R_current_ref, t_current_ref };
   }

   return ret;
}

void convertToKeypoints(const CorrVec& v, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, 
      float size, float angle, float response, int octave, int classid)
{
   unsigned int i = 0;
   for (auto iter = v.begin(); iter != v.end(); iter++, i++) 
   {
      const pair<Point2i,Point2i>& corresp_pts = *iter;
      Point2f&& p1 = Point2f(corresp_pts.first.x,corresp_pts.first.y);
      Point2f&& p2 = Point2f(corresp_pts.second.x,corresp_pts.second.y);
      kpts1[i] = KeyPoint(p1, size, angle, response, octave, classid);
      kpts2[i] = KeyPoint(p2, size, angle, response, octave, classid);
   }
}
void ImageSeries::sortPoints(CorrVec& v)
{

   sort(v.begin(), v.end(), [](pair<Point2i,Point2i> p1, pair<Point2i,Point2i> p2) 
         {
         if (p1.first.x < p2.first.x) return true;
         else if (p1.first.x == p2.first.x) return p1.first.y < p2.first.y;
         else return false;
         }
       );
}

void ImageSeries::show_matches(unsigned int indexl, unsigned int indexr, const CorrVec& v) const
{      
      const unsigned int n = v.size();
      vector<KeyPoint> kpts1(n), kpts2(n);
      vector<DMatch> matches(n);
      convertToKeypoints(v, kpts1, kpts2);
      for (unsigned int i = 0; i < n; ++i) matches[i] = DMatch(i,i,0);

      Mat matchesImg;
      drawMatches(mImages[indexl],kpts1,mImages[indexr],kpts2,matches,matchesImg);
      namedWindow("Foobar", WINDOW_NORMAL);
      imshow("Foobar", matchesImg);
      waitKey(0);
}

ImageSeries::ImageSeries(Mat&& first_frame, Mat&& second_frame, Mat&& reference,
      Mat&& camera_matrix, Mat&& dist_coeffs)
{
   if (!(first_frame.data && second_frame.data && reference.data))
      throw std::invalid_argument("At least one image has no data.");
   mImages = vector<Mat>(3);
   mImages[0] = first_frame;
   mImages[1] = second_frame;
   mImages[2] = reference;
   mCorrespondences = std::vector<CorrVec>(3);
   mCameraMatrix = camera_matrix;
   mDistCoeffs = dist_coeffs;
}

ImageSeries::ImageSeries(vector<Mat> mats)
{
   if (mats.size() < 3) 
      throw invalid_argument("Too few elements. Must be > 3");
   mImages = move(mats);
}

const Mat& ImageSeries::first_frame(void) const
{
   return mImages[0];
}

const Mat& ImageSeries::second_frame(void) const
{
   return mImages[1];
}

const Mat& ImageSeries::reference_frame(void) const
{
   return mImages[2];
}

const vector<Mat> ImageSeries::frames(void) const
{
   return std::vector<cv::Mat>(mImages.begin() + 3, mImages.end());
}

void ImageSeries::set_images(std::vector<cv::Mat> imgs)
{
   mImages.insert(mImages.end(), imgs.begin(), imgs.end());
   mCorrespondences.reserve(mImages.size());

}
void ImageSeries::add_image(cv::Mat img)
{
   mImages.push_back(img);
}

void ImageSeries::add_image(cv::Mat img, CorrVec v)
{
   add_image(img);
   add_correspondences(mImages.size() - 3, v); // works because ctor guarantees size >= 3
}

void ImageSeries::add_correspondences(ImageRole role, CorrVec v)
{
   sortPoints(v);
   mCorrespondences[role] = v;
}

void ImageSeries::add_correspondences(unsigned int index, CorrVec v)
{
   if (index + 3 < mImages.size()) 
   {
      sortPoints(v);
      if (index + 3 >= mCorrespondences.size())
      {
         mCorrespondences.resize(index + 3 + 1);
      }
      mCorrespondences[index + 3] = v;
      /* show_matches(0,index+3,mCorrespondences[index + 3]); */
   } else throw out_of_range("There is no such intermediate image.");
}

const CorrVec& ImageSeries::correspondences_for_frame(ImageRole role) const
{
   if (mCorrespondences.size() > (unsigned int)role) return mCorrespondences[role];
   else throw out_of_range("There are no correspondences for this role.");
}

const CorrVec& ImageSeries::correspondences_for_frame(unsigned int index) const
{
   if (index + 3 < mCorrespondences.size())
   {
      return mCorrespondences[index + 3];
   } else throw out_of_range("There is no such correspondence vector.");
}

const cv::Mat& ImageSeries::image_for_index(unsigned int index) const
{
   if (index + 3 <= mImages.size()) return mImages[index + 3];
   else throw out_of_range("No such image.");
}
const cv::Mat& ImageSeries::camera_matrix(void) const
{
   return mCameraMatrix;
}
const cv::Mat& ImageSeries::dist_coeffs(void) const
{
   return mDistCoeffs;
}
 unsigned int ImageSeries::num_intermediate_imgs() const
{
   return mImages.size() - 3;
}
