#include "stereo_v3.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

static string data_root("/Volumes/Macintosh HD/Users/Rasmus/Downloads/Castle Data/");
static int image_cols = 3072, image_rowss = 2048;

string makeFileName(int i, string subdir, string extension)
{
   char filename[100];
   sprintf(filename, "%s%s/%04d.%s", data_root.c_str(), subdir.c_str(), i, extension.c_str());
   return string(filename);
}

Mat readNumbersFromStream(int rows, int cols, istream& stream)
{
   Mat m(rows, cols, CV_64FC1);
   double d;
   for (int i = 0; i < rows; i++) 
   {
      for (int j = 0; j < cols; j++) 
      {
         stream >> d;
         m.at<double>(i, j) = d;
      }
   }
   return m;
}

Mat readCameraMatrix(int i)
{
   string fileName = makeFileName(i, "camera_matrices", "png.camera");
   ifstream file(fileName);

   return readNumbersFromStream(3, 3, file);
}

Mat readRotationMatrix(int i)
{
   string fileName = makeFileName(i, "camera_matrices", "png.camera");
   ifstream file(fileName);
   string s;
   for (int i = 0; i < 4; i++) 
      getline(file, s);
   return readNumbersFromStream(3,3,file);
}

Mat readTranslationVector(int i)
{
   string fileName = makeFileName(i, "camera_matrices", "png.camera");
   ifstream file(fileName);
   string s;
   for (int i = 0; i < 7; i++) 
      getline(file, s);
   return readNumbersFromStream(3,1,file);
}

int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   Mat left_img, right_img;

   int left_index, right_index;
   left_index = 3, right_index = 4;

   left_img = imread(makeFileName(left_index, "images", "png"), IMREAD_COLOR);
   right_img = imread(makeFileName(right_index, "images", "png"), IMREAD_COLOR);

   if (!left_img.data || !right_img.data)
   {
      cout << "failed to load." << endl;
      return 1;
   }
   
   Mat K = readCameraMatrix(0);
   Mat R1_world = readRotationMatrix(left_index), R2_world = readRotationMatrix(right_index);
   cout << "camera matrix: " << K << endl;
   cout << "R1: " << R1_world << endl;
   cout << "R2: " << R2_world << endl;
   double worldScale;
   Mat img_matches;
   Mat R, t;
   Mat dist_coefficients;
   computePoseDifference(left_img, right_img, args, K, dist_coefficients, worldScale, R, t, img_matches);
   Mat foo, bar;
   Vec3d angles = RQDecomp3x3(R1_world, foo, bar);
   cout << "R1 angles: " << angles << endl;
   angles = RQDecomp3x3(R2_world, foo, bar);
   cout << "R2: " << R2_world << endl;
   cout << "R2 angles: " << angles << endl;

   Mat combined = R.t() * R1_world;
   cout << "R * R1: " << combined << endl;
   cout << "R * R1 angles: " << RQDecomp3x3(combined, foo, bar) << endl;

   if (args.draw_matches) 
   {
      namedWindow("Matches", CV_WINDOW_NORMAL);
      imshow("Matches", img_matches);
      waitKey(0);
   }

   return 0;
}
