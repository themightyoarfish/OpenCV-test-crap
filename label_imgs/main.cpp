#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "DualImageWindow.hpp"

using std::cout;
using std::cerr;
using std::endl;
using namespace imagelabeling;
using namespace cv;

int main(int argc, const char *argv[])
{
   if (argc != 3) 
   {
      cerr << "Gief two imgs pls." << endl;
      return -1;
   }
   Mat left = imread(argv[1]);
   Mat right = imread(argv[2]);
   if (!left.data || !right.data) 
   {
      cerr << "Failed to read at leas one image." << endl;
      return -2;
   }
   DualImageWindow window(left, right);
   window.show();
   return 0;
}
