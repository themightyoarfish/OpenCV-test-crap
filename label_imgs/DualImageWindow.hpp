#include <opencv2/core.hpp>
#include <vector>
using std::string;
using std::vector;
using std::pair;
using cv::Mat;
using cv::Point2i;
namespace imagelabeling 
{
   class DualImageWindow 
   {
      public:
         explicit DualImageWindow (const Mat left_img, const Mat right_img, 
               const string window_name = "Label Images");
         void show();
         static void mouseCallback(int event, int x, int y, int flags, void* data);
         virtual ~DualImageWindow();

      private:
         Mat left_img, right_img, combined_imgs;
         string window_name;
         vector<Point2i> correspondences;
         bool firstPointSet;

         void refresh();
         bool handleKeyEvent(const int key) const;
         void combine_imgs(const Mat& left_img, const Mat& right_img);
   };

} /* namespace imagelabeling */
