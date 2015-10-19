#include "ImageSeries.hpp"
#include <stdexcept>

using namespace std;
using namespace cv;
ImageSeries::ImageSeries(Mat&& first_frame, Mat&& second_frame, Mat&& reference)
{
   if (!(first_frame.data && second_frame.data && reference.data))
      throw std::invalid_argument("At least one image has no data.");
   images = vector<Mat>(3);
   images[0] = first_frame;
   images[1] = second_frame;
   images[2] = reference;
}
ImageSeries::ImageSeries(vector<Mat> mats)
{
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
}
void ImageSeries::add_image(cv::Mat img)
{
   images.push_back(img);
}
