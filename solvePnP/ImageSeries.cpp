#include "ImageSeries.hpp"
#include <stdexcept>

using namespace std;
using namespace cv;

void convertToKeypoints(CorrVec& v, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, 
      float size, float angle, float response, int octave, int classid)
{
   unsigned int i = 0;
   for (auto iter = v.begin(); iter != v.end(); iter++, i++) 
   {
      pair<Point2i,Point2i>& corresp_pts = *iter;
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

void ImageSeries::showMatches(unsigned int indexl, unsigned int indexr, CorrVec& v)
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

Mat& ImageSeries::first_frame(void)
{
   return mImages[0];
}

Mat& ImageSeries::second_frame(void)
{
   return mImages[1];
}

Mat& ImageSeries::reference_frame(void)
{
   return mImages[2];
}

const vector<Mat> ImageSeries::frames(void)
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
      /* showMatches(0,index+3,mCorrespondences[index + 3]); */
   } else throw out_of_range("There is no such intermediate image.");
   std::cout << "Added cores. Size is " << mCorrespondences.size() << std::endl;
}

CorrVec& ImageSeries::correspondences_for_frame(ImageRole role)
{
   if (mCorrespondences.size() > (unsigned int)role) return mCorrespondences[role];
   else throw out_of_range("There are no correspondences for this role.");
}

CorrVec& ImageSeries::correspondences_for_frame(unsigned int index)
{
   if (index + 3 < mCorrespondences.size())
   {
      return mCorrespondences[index + 3];
   } else throw out_of_range("There is no such correspondence vector.");
}

cv::Mat& ImageSeries::image_for_index(unsigned int index)
{
   if (index + 3 <= mImages.size()) return mImages[index + 3];
   else throw out_of_range("No such image.");
}
cv::Mat& ImageSeries::camera_matrix()
{
   return mCameraMatrix;
}
cv::Mat& ImageSeries::dist_coeffs()
{
   return mDistCoeffs;
}
