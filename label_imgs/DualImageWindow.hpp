#include <opencv2/core.hpp>
using std::string;
using cv::Mat;
namespace imagelabeling 
{
   class DualImageWindow 
   {
      public:
         explicit DualImageWindow (Mat left_img, Mat right_img, string window_name = "Label Images");
         void show();
         virtual ~DualImageWindow();

      private:
         Mat left_img, right_img;
         string window_name;
   };

} /* namespace imagelabeling */
