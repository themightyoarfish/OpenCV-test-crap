#include "stereo_v3.hpp"
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static vector<string> filenames = {
   "/Users/Rasmus/Desktop/Bahnhof/ref_corrected.JPG",
   "/Users/Rasmus/Desktop/Bahnhof/2.JPG",
   "/Users/Rasmus/Desktop/Bahnhof/3.JPG",
   "/Users/Rasmus/Desktop/Bahnhof/4.JPG",
   "/Users/Rasmus/Desktop/Bahnhof/5.JPG",
   "/Users/Rasmus/Desktop/Bahnhof/first_frame_centered.JPG",
};

tuple<vector<Point2f>, vector<Point2f>> readPtsFromFile(string filename)
{
   vector<Point2f> imgpts1, imgpts2;
   ifstream file(filename, ios::in);
   char line[100];
   bool first_part = true;
   while (!file.eof()) 
   {
      file.getline(line, 30);
      if (file.eof()) break;
      cout << line << endl;
      if (line[0] == '"') continue;
      if(line[0] == '\0') first_part = false;
      Point2f p;
      sscanf(line, "%f,%f", &p.x, &p.y);
      if (first_part) 
         imgpts1.push_back(p);
      else
         imgpts2.push_back(p);
   }
   return make_tuple(imgpts1, imgpts2);
}

#define GET_BASE_NAME(file)\
{\
   size_t last_backslash = file.find_last_of("/");\
   size_t ext_start = file.find_last_of(".");\
   file = file.substr(last_backslash + 1, ext_start - last_backslash - 1);\
}

int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);
   for (int i = 0; i < filenames.size() -1; i++) 
   {
      vector<Point2f> imgpts1, imgpts2;
      char filename[100];

      string f1 = filenames[i];
      GET_BASE_NAME(f1);
      string f2 = filenames.back();
      GET_BASE_NAME(f2);
      sprintf(filename, "/Users/Rasmus/Desktop/Bahnhof/imgpts_%s->%s.txt",f1.c_str(),f2.c_str());
      tie(imgpts1, imgpts2) = readPtsFromFile(filename);
   }
   return 0;
}
