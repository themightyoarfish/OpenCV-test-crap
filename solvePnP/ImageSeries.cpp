#include "ImageSeries.hpp"
#include <algorithm>
#include <stdexcept>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

const Point2f INVALID_PT = Point2f(-1,-1);

vector<Point2f> remove_invalid(vector<Point2f>& v)
{
   vector<Point2f> ret;
   for (auto& p : v)
      if(p != INVALID_PT) ret.push_back(p);
   return ret;
}
vector<Point2f> operator& (vector<Point2f>& v, vector<Point2f>& mask)
{
   if (v.size() != mask.size())
      throw std::invalid_argument("Vectors mus have same length. (Was " + std::to_string(v.size())+ " and " + std::to_string(mask.size()) + ")");
   vector<Point2f> ret(v.size());
   for (int i = 0; i < v.size(); ++i)
   {
      if (v[i] == INVALID_PT or mask[i] == INVALID_PT)
         ret[i] = INVALID_PT;
      else 
         ret[i] = v[i];
   }
   return ret;
}
std::tuple<vector<Point2f>, vector<Point2f>> matches_to_points(vector<DMatch>& matches, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2)
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

vector<PoseData> runEstimate(const ImageSeries& series, bool interactive, unsigned int resize_factor, bool autofeatures)
{

   unsigned int n1, n2, nref;
   auto match_comparator = [](DMatch& m1, DMatch& m2) { return m1.distance < m2.distance; };
   auto scale_pt_up = [&resize_factor](Point2f& p) { return p * (float)resize_factor; };
   vector<KeyPoint> kpts_first, kpts_second, kpts_ref;
   vector<Point2f>  pts_first,  // all ff kpts converted to points
      pts_second, pts_ref; // all 2nd and ref frame kpts with matches in ff, all others are INVALID_PT
   Mat descriptors_first, descriptors_second, descriptors_ref;
   Ptr<Feature2D> detector;
   detector = AKAZE::create();
   Mat first_frame, second_frame, reference_frame;
   const float RATIO = 0.8;

   BFMatcher matcher;
   vector<DMatch> matches_first_second, matches_first_ref; // when autodetecting, hold the matches

   if (not autofeatures)
   {
      const CorrVec& corr_first_second = series.correspondences_for_frame(ImageSeries::SECOND_FRAME);
      const CorrVec& corr_first_ref    = series.correspondences_for_frame(ImageSeries::REF_FRAME);
      n1 = n2 = nref                   = corr_first_second.size();
      convertToKeypoints(corr_first_second, kpts_first, kpts_second);
      convertToKeypoints(corr_first_ref   , kpts_first, kpts_ref);
      for (unsigned int i = 0; i < n1; ++i) 
      {
         pts_first.push_back(kpts_first[i].pt);
         pts_second.push_back(kpts_second[i].pt);
         pts_ref.push_back(kpts_ref[i].pt);
      }
   } else 
   {

      resize(series.first_frame(),      first_frame,      Size(),  1. / resize_factor,  1. / resize_factor);
      resize(series.second_frame(),     second_frame,     Size(),  1. / resize_factor,  1. / resize_factor);
      resize(series.reference_frame(),  reference_frame,  Size(),  1. / resize_factor,  1. / resize_factor);

      detector->detectAndCompute(first_frame,      noArray(),  kpts_first,   descriptors_first);
      detector->detectAndCompute(second_frame,     noArray(),  kpts_second,  descriptors_second);
      detector->detectAndCompute(reference_frame,  noArray(),  kpts_ref,     descriptors_ref);

      n1 = kpts_first.size();
      n2 = kpts_second.size();
      nref = kpts_ref.size();
      std::transform(kpts_first.begin(), kpts_first.end(), pts_first.begin(), [&](KeyPoint& k) { return k.pt; });
      pts_second.resize(n1, Point2f(-1, -1));
      pts_ref.resize(n1, Point2f(-1, -1));

      vector<vector<DMatch>>  candidates_first_second,  candidates_first_ref;
      matcher = BFMatcher(NORM_HAMMING);
      matcher.knnMatch(descriptors_first, descriptors_second, candidates_first_second, 2);
      matcher.knnMatch(descriptors_first, descriptors_ref   , candidates_first_ref   , 2);

      for (int i = 0; i < candidates_first_second.size(); i++)
      {
         DMatch& m1 = candidates_first_second[i][0];
         DMatch& m2 = candidates_first_second[i][1];
         if (m1.distance < RATIO * m2.distance)
            matches_first_second.push_back(m1);
      }
      for (int i = 0; i < candidates_first_ref.size(); i++)
      {
         DMatch& m1 = candidates_first_ref[i][0];
         DMatch& m2 = candidates_first_ref[i][1];
         if (m1.distance < RATIO * m2.distance)
            matches_first_ref.push_back(m1);
      }

      std::sort(matches_first_second.begin(), matches_first_second.end(), match_comparator);
      std::sort(matches_first_ref.begin(), matches_first_ref.end(), match_comparator);

      for (DMatch& m : matches_first_second)
         pts_second[m.queryIdx] = kpts_second[m.trainIdx].pt;
      for (DMatch& m : matches_first_ref)
         pts_ref[m.queryIdx] = kpts_ref[m.trainIdx].pt;

      /* std::tie(pts_first, pts_second) = matches_to_points(matches_first_second, kpts_first, kpts_second); */
      /* std::tie(pts_first, pts_ref)       = matches_to_points(matches_first_ref   , kpts_first, kpts_ref); */

      if (resize_factor > 1)
      {
         std::transform(pts_first.begin() , pts_first.end() , pts_first.begin() , scale_pt_up);
         std::transform(pts_second.begin(), pts_second.end(), pts_second.begin(), scale_pt_up);
         std::transform(pts_ref.begin()   , pts_ref.end()   , pts_ref.begin()   , scale_pt_up);
      }
   }

   Mat_<double> camera_matrix = series.camera_matrix();
   Mat dist_coeffs = series.dist_coeffs();

   double focal = camera_matrix(0,0);
   Point2d principalPoint(camera_matrix(0,2), camera_matrix(1,2));

   std::vector<Point2f> good_pts_first = pts_first & pts_second,
      good_pts_second                  = pts_second & pts_first;

   /** DRAW THE MATCHES */
   /* vector<KeyPoint> k1(good_pts_first.size()), k2(good_pts_second.size()); */
   /* vector<DMatch> matches(good_pts_first.size()); */
   /* for (int i = 0; i < good_pts_first.size(); ++i) */
   /* { */
   /*    matches[i] = DMatch(i,i,0); */
   /* } */
   /* std::transform(good_pts_first.begin(), good_pts_first.end(), k1.begin(), [](Point2f& p) { return KeyPoint(p,4); }); */
   /* std::transform(good_pts_second.begin(), good_pts_second.end(), k2.begin(), [](Point2f& p) { return KeyPoint(p,4); }); */
   /* Mat matchesImg; */
   /* drawMatches(series.first_frame(),k1,series.second_frame(),k2,matches,matchesImg); */
   /* namedWindow("Foobar", WINDOW_NORMAL); */
   /* imshow("Foobar", matchesImg); */
   /* waitKey(0); */
   /* /1* ================= *1/ */
   Mat mask; // TODO: Type as _<uchar>
   Mat E = findEssentialMat(remove_invalid(good_pts_first), remove_invalid(good_pts_second), focal, principalPoint, RANSAC, 0.999, 1, mask);

   Mat R, t;
   int inliers = recoverPose(E, remove_invalid(good_pts_first), remove_invalid(good_pts_second), R, t, focal, principalPoint, mask);
   Vec3d angles = rotationMatToEuler(R);

   // figure out which points in the good_pts_* arrays are affected by the mask
   int c = 0;
   for (unsigned int i = 0; i < n1; ++i)
      if (good_pts_first[i] != INVALID_PT )
         if (mask.at<uchar>(c++,0) == 0) good_pts_first[i] = INVALID_PT;
   c = 0;
   for (unsigned int i = 0; i < n2; ++i)
      if (good_pts_second[i] != INVALID_PT )
         if (mask.at<uchar>(c++,0) == 0) good_pts_second[i] = INVALID_PT;


   /* vector<Point2f> pts_first_masked, pts_second_masked, pts_ref_masked; */
   /* for (int i = 0; i < n1; i++) */ 
   /* { */
   /*    if (mask.at<uchar>(i,0) == 1) */ 
   /*    { */
   /* pts_first_masked.push_back(pts_first[i]); */
   /* pts_second_masked.push_back(pts_second[i]); */
   /* pts_ref_masked.push_back(pts_ref[i]); */
   /* } */ 
   /* } */

   Mat pnts4D;
   Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
   Mat p2[2] = { R, t }; 
   hconcat(p2, 2, P2);
   P2 = camera_matrix * P2;
   triangulatePoints(P1, P2, remove_invalid(good_pts_first), remove_invalid(good_pts_second), pnts4D);
   pnts4D = pnts4D.t();
   Mat dehomogenized;
   convertPointsFromHomogeneous(pnts4D, dehomogenized);
   dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols

   /*** SOLVEPNP ***/
   Mat rvec, t_first_ref; // note the indices are reversed compared to my thesis
   Mat R_first_ref;
   vector<Point2f> pts_ref_usable = pts_ref & good_pts_first;
   Mat dehomogenized_good_subset(remove_invalid(pts_ref_usable).size(), 3, dehomogenized.type());

   int c1 = 0, c2 = 0;
   for (int i = 0; i < good_pts_first.size(); ++i)
   {
      if (good_pts_first[i] != INVALID_PT)
      {
         if (pts_ref[i] != INVALID_PT)
         {
            dehomogenized_good_subset.at<double>(c2,0) = dehomogenized.at<double>(c1,0);
            dehomogenized_good_subset.at<double>(c2,1) = dehomogenized.at<double>(c1,1);
            dehomogenized_good_subset.at<double>(c2,2) = dehomogenized.at<double>(c1,2);
            c2++;
         }
         c1++;
      }
   }

   solvePnP(dehomogenized_good_subset, remove_invalid(pts_ref_usable), camera_matrix, noArray(), rvec, t_first_ref);
   Rodrigues(rvec,R_first_ref);

   std::cout << "R_first_ref: " << rotationMatToEuler(R_first_ref) << std::endl;
   std::cout << "t_first_ref: " << t_first_ref << std::endl;

   const unsigned int numImgs = series.num_intermediate_imgs();
   vector<PoseData> ret(numImgs);

   for (unsigned int i = 0; i < numImgs; ++i) 
   {
      vector<Point2f> good_pts_current;
      vector<KeyPoint> kpts_current;
      if (not autofeatures)
      {
         CorrVec corr_first_current = series.correspondences_for_frame(i);
         convertToKeypoints(corr_first_current, kpts_first, kpts_current);
         for(KeyPoint& k : kpts_current)
            good_pts_current.push_back(k.pt);
         good_pts_current = good_pts_current & good_pts_first;
      } else 
      {
         /* vector<KeyPoint> kpts_current; */
         /* Mat descriptors_current; */
         /* Mat current_frame; */
         /* resize(series.series.image_for_index(i), current_frame, Size(),  1. / resize_factor,  1. / resize_factor); */
         /* detector->detectAndCompute(current_frame, noArray(), kpts_current, descriptors_current); */

         /* vector<DMatch> matches_first_current; */
         /* vector<vector<DMatch>>  candidates_first_current; */
         /* matcher.knnMatch(descriptors_first, descriptors_current, candidates_first_current, 2); */

         /* for (int i = 0; i < candidates_first_current.size(); i++) */
         /* { */
         /*    DMatch& m1 = candidates_first_current[i][0]; */
         /*    DMatch& m2 = candidates_first_current[i][1]; */
         /*    if (m1.distance < RATIO * m2.distance) */
         /*       matches_first_current.push_back(m1); */
         /* } */
         /* std::sort(matches_first_current.begin(), matches_first_current.end(), match_comparator); */

         /* for (DMatch& m : matches_first_current) */
         /*    pts_current[m.queryIdx] = kpts_current[m.trainIdx].pt; */
      }
      vector<Point2f> pts_current_usable = good_pts_current;
      Mat dehomogenized_good_subset(remove_invalid(pts_current_usable).size(), 3, dehomogenized.type());

      int c1 = 0, c2 = 0;
      for (int i = 0; i < good_pts_first.size(); ++i)
      {
         if (good_pts_first[i] != INVALID_PT)
         {
            if (pts_ref[i] != INVALID_PT)
            {
               dehomogenized_good_subset.at<double>(c2,0) = dehomogenized.at<double>(c1,0);
               dehomogenized_good_subset.at<double>(c2,1) = dehomogenized.at<double>(c1,1);
               dehomogenized_good_subset.at<double>(c2,2) = dehomogenized.at<double>(c1,2);
               c2++;
            }
            c1++;
         }
      }
      Mat R_first_current, t_first_current;
      solvePnP(dehomogenized_good_subset, remove_invalid(good_pts_current), camera_matrix, noArray(), rvec, t_first_current);
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
