#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char *argv[])
{
   using namespace cv;
   using namespace std;

   if (argc != 2)
   {
      cout << "USE: " << argv[0] << " <file_path>" << endl;
      return 1;
   }

   //Our color image
   Mat imageMat = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   if (imageMat.empty())
   {
      cerr << "ERROR: Could not read image " << argv[1] << endl;
      return 1;
   }
   Mat transparent( imageMat.rows, imageMat.cols, CV_8UC4);
   cout << "imageMat: " << imageMat.channels() << ", " << imageMat.depth() << endl;
   cout << "transparent: " << transparent.channels() << ", " << transparent.depth() << endl;
   Mat srcImg[] = {imageMat, imageMat};
   int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
   mixChannels( srcImg, 2, &transparent, 1, from_to, 4 );
   imwrite("out.png", transparent);

   return 0;
}
