#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include "DualImageWindow.hpp"
#include "serialization.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ofstream;
using std::vector;
using std::pair;
using std::ios;
using boost::filesystem::path;
using namespace imagelabeling;
using namespace cv;

vector<pair<Point2i,Point2i>> label_images(const Mat& left, const Mat& right)
{
   DualImageWindow window(left, right);
   window.show();
   return window.points();
}

int main(int argc, const char *argv[])
{
   if (argc < 3) 
   {
      cerr << "Gief two imgs pls." << endl;
      return -1;
   }
   path path_left(argv[1]), path_right(argv[2]);
   string fname = argc >= 4 ? fname = string(argv[3]) : path_left.filename().string() 
                                                      + "->" 
                                                      + path_right.filename().string() 
                                                      + "_pts";
   Mat left = imread(argv[1]);
   Mat right = imread(argv[2]);
   if (!left.data || !right.data) 
   {
      cerr << "Failed to read at least one image." << endl;
      return -2;
   }
   auto points = label_images(left, right);
   serialize_vector(points, fname);
   points = deserialize_vector<Point2i,Point2i>(fname);
   for (auto iter = points.begin() ; iter != points.end() ; iter++)
      cout << "(" << iter->first << "," << iter->second << ")"<< endl;
   return 0;
}
