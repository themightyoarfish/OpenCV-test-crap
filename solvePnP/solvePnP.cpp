#include <opencv2/opencv.hpp>
#include <iostream>
#include <tclap/CmdLine.h>
#include <fstream>
#include <prettyprint/prettyprint.hpp>
#include "ImageSeries.hpp"

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
   ValueArg<string> images_arg("i", 
         "image-names",
         "Filenames of all images. Should be given as a path to a\
         file with newline-separated filenames. The first frame must\ 
         come first, the second frame second, the reference frame last.",
         true,
         "n/a",
         "File listing all image filenames");
   cmd.add(images_arg);

   ValueArg<string> correspondences_arg("c", 
         "correspondences",
         "Filenames of all files containing the correspondences. Should be given as a path to a\
         file with newline-separated filenames.\
         The line i will contain the matches between first frame\
         and image i from the image file list, excluding the first frame itself.",
         true,
         "n/a",
         "File listing all correspondence filenames");

   cmd.add(correspondences_arg);
   cmd.parse(argc, argv);

   vector<string> image_filenames = getLinesFromFile(images_arg.getValue(), [](string s){ return (bool)s.length(); });
   vector<string> correspondence_filenames = getLinesFromFile(correspondences_arg.getValue());
   if (image_filenames.size() != correspondence_filenames.size() + 1) 
   {
      cerr << "There must be one less correspondence filename than there are image file names." << endl;
      return -2;
   }
   if (image_filenames.size() <= 3)
   {
      cerr << "Error. Must be at least 4 files." << endl;
      return -1;
   }

   try
   {
      ImageSeries series(
            imread(image_filenames[0]), 
            imread(image_filenames[1]),
            imread(image_filenames.back())
            );
      for (auto iter = image_filenames.begin() + 2; iter != image_filenames.end() - 1; iter++) 
      {
         series.add_image(imread(*iter));
      }
   } catch (std::exception& e)
   {
      cerr << "Caught exception: " << e.what() << endl;
      return -2;
   }

   return 0;
}
