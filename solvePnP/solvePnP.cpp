#include <opencv2/opencv.hpp>
#include <iostream>
#include <tclap/CmdLine.h>
#include <fstream>
#include <prettyprint/prettyprint.hpp>

using namespace cv;
using namespace std;

vector<string> getLinesFromFile(const string& fname, bool (*yesno)(string) = [](string s) { return true; })
{
   ifstream file(fname);
   vector<string> vec;
   string current_line;
   while (getline(file, current_line)) 
      if (yesno(current_line)) vec.push_back(current_line);
   return vec;
}

int main(int argc, const char *argv[])
{
   using TCLAP::CmdLine;
   using TCLAP::ValueArg;
   CmdLine cmd("Useful message", ' ', "0.1");
   ValueArg<string> images("i", 
         "image-names",
         "Filenames of all images. Should be given as a path to a\
         file with newline-separated filenames. The first frame must\ 
         come first, the second frame second, the reference frame last.",
         true,
         "n/a",
         "File listing all image filenames");
   cmd.add(images);

   cmd.parse(argc, argv);

   vector<string> image_filenames = getLinesFromFile(images.getValue(), [](string s){ return (bool)s.length(); });
   cout << image_filenames << endl;

   return 0;
}
