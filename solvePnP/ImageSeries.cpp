#include "ImageSeries.hpp"
#include <stdexcept>

using namespace std;
using namespace cv;
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
{      /* *** Draw the matches for debugging ** */
      const unsigned int n = v.size();
      vector<KeyPoint> kpts1(n), kpts2(n);
      vector<DMatch> matches(n);
      unsigned int i = 0;
      for (auto iter = v.begin(); iter != v.end(); iter++, i++) 
      {
         pair<Point2i,Point2i>& corresp_pts = *iter;
         Point2f&& p1 = Point2f(corresp_pts.first.x,corresp_pts.first.y);
         Point2f&& p2 = Point2f(corresp_pts.second.x,corresp_pts.second.y);
         kpts1[i] = KeyPoint(p1, 10);
         kpts2[i] = KeyPoint(p2, 10);
         matches[i] = DMatch(i,i,0);
      }

      Mat matchesImg;
      drawMatches(images[indexl],kpts1,images[indexr],kpts2,matches,matchesImg);
      namedWindow("Foobar", WINDOW_NORMAL);
      imshow("Foobar", matchesImg);
      waitKey(0);

         /* ************************************* */

}
ImageSeries::ImageSeries(Mat&& first_frame, Mat&& second_frame, Mat&& reference)
{
   if (!(first_frame.data && second_frame.data && reference.data))
      throw std::invalid_argument("At least one image has no data.");
   images = vector<Mat>(3);
   images[0] = first_frame;
   images[1] = second_frame;
   images[2] = reference;
   correspondences = std::vector<CorrVec>(3);
}
ImageSeries::ImageSeries(vector<Mat> mats)
{
   if (mats.size() < 3) 
      throw invalid_argument("Too few elements. Must be > 3");
   images = move(mats);
}
Mat& ImageSeries::first_frame(void)
{
   return images[0];
}
Mat& ImageSeries::second_frame(void)
{
   return images[1];
}
Mat& ImageSeries::reference_frame(void)
{
   return images[2];
}
const vector<Mat> ImageSeries::frames(void)
{
   return std::vector<cv::Mat>(images.begin() + 3, images.end());
}
void ImageSeries::set_images(std::vector<cv::Mat> imgs)
{
   images.insert(images.end(), imgs.begin(), imgs.end());
   correspondences.reserve(images.size());
}
void ImageSeries::add_image(cv::Mat img)
{
   images.push_back(img);
}
void ImageSeries::add_image(cv::Mat img, CorrVec v)
{
   add_image(img);
   add_correspondences(images.size() - 3, v); // works because ctor guarantees size >= 3
}
void ImageSeries::add_correspondences(ImageRole role, CorrVec v)
{
   sortPoints(v);
   correspondences[role] = v;
}
void ImageSeries::add_correspondences(unsigned int index, CorrVec v)
{
   if (index + 3 <= images.size()) 
   {
      sortPoints(v);
      /* showMatches(0,index+3,v); */
      correspondences[index] = v;
   } else throw out_of_range("There is no such intermediate image.");
}
CorrVec& ImageSeries::correspondences_for_frame(ImageRole role)
{
   return correspondences_for_frame((unsigned int)role - 3);
}
CorrVec& ImageSeries::correspondences_for_frame(unsigned index)
{
   if (index + 3 <= correspondences.size())
   {
      return correspondences[index + 3];
   } else throw out_of_range("There is no such correspondence vector.");
}
cv::Mat& ImageSeries::image_for_index(unsigned int index)
{
   if (index + 3 <= images.size()) return images[index + 3];
   else throw out_of_range("No such image.");
}
