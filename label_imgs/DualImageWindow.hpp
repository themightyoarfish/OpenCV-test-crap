#include <opencv2/core.hpp>
#include <vector>

typedef std::pair<cv::Point2i,cv::Point2i> PointPair;
namespace imagelabeling 
{
   class DualImageWindow 
   {
      public:
         explicit DualImageWindow (const cv::Mat left_img, const cv::Mat right_img, std::vector<PointPair> initial_points = std::vector<PointPair>(),
               const std::string window_name = "Label Images");
         std::vector<std::pair<cv::Point2i,cv::Point2i>> points();
         void show();


         static void mouseCallback(int event, int x, int y, int flags, void* data);
         virtual ~DualImageWindow();

      private:
         cv::Mat left_img, right_img, combined_imgs;
         std::string window_name;
         std::vector<cv::Point2i> correspondences;
         bool firstPointSet;

         void refresh();
         bool handleKeyEvent(const int key);
         void combine_imgs(const cv::Mat& left_img, const cv::Mat& right_img);
   };

} /* namespace imagelabeling */
