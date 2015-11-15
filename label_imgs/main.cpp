#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <tclap/CmdLine.h>
#include "DualImageWindow.hpp"
#include "serialization.hpp"

using std::cout;
using std::cin;
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

bool confirm_action(const string& prompt)
{
   char response;
   do {
      cout << prompt << " [y/n] ";
      cout.flush();
      cin >> response;
   } while (response != 'y' && response != 'n');
   return response == 'y';
}

vector<pair<Point2i,Point2i>> label_images(const Mat& left, const Mat& right, const string& window_name, const string& initial_pts = "")
{
   DualImageWindow window(left, right, initial_pts.empty() ? vector<PointPair>() : deserialize_vector<Point2i,Point2i>(initial_pts), window_name);
   window.show();
   return window.points();
}

int main(int argc, const char *argv[])
{
   using TCLAP::CmdLine;
   using TCLAP::ValueArg;
   CmdLine cmd("", ' ', "0.1");
   ValueArg<string> left_img_arg(
         "l", "left-image",
         "An image file with a common format (one understood by OpenCV)",
         true,
         "n/a",
         "file"
         );
   ValueArg<string> right_img_arg(
         "r", "right-image",
         "An image file with a common format (one understood by OpenCV)",
         true,
         "n/a",
         "file"
         );
   cmd.add(left_img_arg);
   cmd.add(right_img_arg);

   ValueArg<string> output_arg(
         "o", "output",
         "The file to which the corresponding points will be serialized. They can be read back with deserialize_vector() from serialization.hpp",
         false,
         "",
         "file"
         );
   cmd.add(output_arg);

   ValueArg<string> correspondence_arg(
         "c",
         "correspondences",
         "File with correspondences between the two images. This can be used to prepopulate the display and update correspondences. The file must have the format used by serialize_vector from serialization.hpp",
         false,
         "",
         "file"
         );
   cmd.add(correspondence_arg);

   cmd.parse(argc, argv);

   path path_left(left_img_arg.getValue()), path_right(right_img_arg.getValue());
   string initial_pts_file = correspondence_arg.getValue();
   string out_fname = output_arg.getValue();

   if (out_fname.empty())
      out_fname = path_left.filename().string() 
         + "->" 
         + path_right.filename().string() 
         + "_pts";


   Mat left = imread(path_left.string());
   Mat right = imread(path_right.string());
   if (!left.data || !right.data) 
   {
      cerr << "Failed to read at least one image." << endl;
      return -2;
   }
   auto points = label_images(left, right, path_left.filename().string() + " -> " + path_right.filename().string(), initial_pts_file);
   path out_path(out_fname);
   if (!points.empty())
   {
      if (boost::filesystem::exists(out_path))
      {
         if (confirm_action("File already exists. Overwrite?")) serialize_vector(points, out_fname);
      } else serialize_vector(points, out_fname);

   }
   return 0;
}
