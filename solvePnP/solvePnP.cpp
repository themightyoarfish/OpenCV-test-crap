#include <opencv2/opencv.hpp>
#include <iostream>
#include <tclap/CmdLine.h>
#include <fstream>
#include <prettyprint/prettyprint.hpp>
#include "ImageSeries.hpp"
#include "serialization.hpp"
#include "CalibrationFileReader.h"
#include "utils.h"

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
   using TCLAP::SwitchArg;
   CmdLine cmd("Useful message", ' ', "0.1");
   ValueArg<string> images_arg("i", 
         "image-names",
         "Filenames of all images. Should be given as a path to a\
         file with newline-separated filenames. The first frame must\
         come first, the second frame second, the reference frame third.",
         true,
         "n/a",
         "File listing all image filenames");
   cmd.add(images_arg);

   ValueArg<string> correspondences_arg("c", 
         "correspondences",
         "Filenames of all files containing the correspondences. Should be given as a path to a\
         file with newline-separated filenames.\
         The line i will contain the filename for matches between first frame\
         and image i from the image file list, excluding the first frame itself.",
         true,
         "n/a",
         "File listing all correspondence filenames");
   cmd.add(correspondences_arg);

   ValueArg<string> calibration_arg("d",
         "calibration-data",
         "Filename of an OpenCV XML storage file with 'Camera_Matrix' and 'Distortion_Coefficients' nodes.",
         false,
         "n/a",
         "Calibration data file"
         );
   cmd.add(calibration_arg);

   SwitchArg interactive_arg("s",
         "show-matches",
         "Flag to indicate whether the matches between each pair of images should be shown.",
         false
         );
   cmd.add(interactive_arg);

   cmd.parse(argc, argv);

   CalibrationFileReader reader(calibration_arg.getValue());
   vector<string> image_filenames = getLinesFromFile(
         images_arg.getValue(),
         [](string s){ return (bool)s.length(); /* remove empty lines */}
         );
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
            imread(image_filenames[2]),
            reader.getCameraMatrix(),
            reader.getDistortionCoeffs()
            );
      std::cout << "Addding image " << image_filenames[0] << std::endl;
      std::cout << "Addding image " << image_filenames[1] << std::endl;
      std::cout << "Addding image " << image_filenames[2] << std::endl;
      for (auto iter = image_filenames.begin() + 3; iter != image_filenames.end(); iter++) 
      {
         std::cout << "Adding image " << *iter << std::endl;
         series.add_image(imread(*iter));
      }
      for (unsigned int i = 0; i < correspondence_filenames.size(); ++i) 
      {
         CorrVec&& corr = deserialize_vector<Point2i,Point2i>(correspondence_filenames[i]);
         std::cout << "Processing correspondences " << correspondence_filenames[i] << std::endl;
         switch (i)
         {
            case 0:
               series.add_correspondences(ImageSeries::SECOND_FRAME, corr); break;
            case 1:
               series.add_correspondences(ImageSeries::REF_FRAME, corr); break;
            default:
               series.add_correspondences(i-2, corr); break;
               break;
         }
      }
      std::cout << "Starting estimation..." << std::endl;
      vector<PoseData> poses = runEstimate(series, interactive_arg.getValue());
      for (auto& i : poses)
      {
         cout << "R: " << rotationMatToEuler(i.R) << endl;
         cout << "t: " << i.t << endl;
      }
   } catch (std::exception& e)
   {
      cerr << "Caught exception: " << e.what() << endl;
      return -2;
   }

   return 0;
}
