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

   Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   //Our color image
   if (src.empty())
   {
      cerr << "ERROR: Could not read image " << argv[1] << endl;
      return 1;
   }
   cout << "Type of src before blur: " << src.type() << endl;
   // noise reduction prior to edge detection; TODO: adjust kernel size?
   blur(src, src, Size(3,3));
   cout << "Type of src after blur: " << src.type() << ", num channels: " << src.channels() << endl;
   imwrite("blurred_image.png", src);

   // detect edges
   int low = 110;
   int high = 3 * low;
   int kernelSz = 3;

   Canny( src, src, low, high, kernelSz );
   cout << "Type of src after Canny: " << src.type() << ", num channels: " << src.channels() << endl;
   imwrite("edge_image.png", src);
   cvtColor(src, src, CV_GRAY2RGB);
   cout << "Type of src after conversion: " << src.type() << endl;
   Mat transparent(src.rows,src.cols, CV_8UC4);
   /* cout << "src: " << src.channels() << ", " << src.depth() << endl; */
   /* cout << "transparent: " << transparent.channels() << ", " << transparent.depth() << endl; */
   Mat srcImg[] = {src, src};
   int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
   mixChannels( srcImg, 2, &transparent, 1, from_to, 4 );
   imwrite("edges.png", transparent);

   return 0;
}
