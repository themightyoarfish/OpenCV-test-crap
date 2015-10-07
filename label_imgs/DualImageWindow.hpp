#include <opencv2/core.hpp>
#include <map>
using std::string;
using std::map;
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
         void refresh();
         Mat left_img, right_img, combined_imgs;
         string window_name;
         map<Point2i,Point2i> correspondences;
   };

} /* namespace imagelabeling */
