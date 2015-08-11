#include "stereo_v3.hpp"
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static vector<string> filenames = {
   "../Data/Bahnhof/ref_corrected.JPG",
   "../Data/Bahnhof/2.JPG",
   "../Data/Bahnhof/3.JPG",
   "../Data/Bahnhof/4.JPG",
   "../Data/Bahnhof/5.JPG",
   "../Data/Bahnhof/first_frame_centered.JPG",
};

tuple<vector<Point2f>, vector<Point2f>> readPtsFromFile(string filename)
{
   vector<Point2f> imgpts1, imgpts2;
   ifstream file(filename, ios::in);
   if (!file.is_open()) 
   {
      cerr << "Wtf, dude." << endl;
      return make_tuple(vector<Point2f>(), vector<Point2f>());
   }
   char line[100];
   bool first_part = true;
   while (!file.eof()) 
   {
      file.getline(line, 30);
      if (file.eof()) break;
      if (line[0] == '"') continue;
      if(line[0] == '\0')
      {
         first_part = false;
         continue;
      }
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

   Mat_<double> camera_matrix(3,3);
   camera_matrix << 
      2.9880631668863380e+03, 0.,                     1.6315000000000000e+03, 
      0.,                     2.9880631668863380e+03, 1.2235000000000000e+03, 
      0.,                     0.,                     1.;
   Mat_<double> dist_coefficients(5,1);
   dist_coefficients << 
      1.4422094911174704e-01, -5.4684174329780899e-01,
      -7.5857781243513097e-04, 1.1949279901859115e-03,
      7.9061044687285797e-01;

   for (int i = 0; i < filenames.size() -1; i++) 
   {
      vector<Point2f> imgpts1, imgpts2;
      char filename[100];

      string f1 = filenames[i];
      GET_BASE_NAME(f1);
      string f2 = filenames.back();
      GET_BASE_NAME(f2);
      sprintf(filename, "../Data/Bahnhof/imgpts_%s->%s.txt",f1.c_str(),f2.c_str());
      cout << "Reading from file " << filename << endl;
      tie(imgpts1, imgpts2) = readPtsFromFile(filename);

      Mat img1 = imread(filenames[i], IMREAD_COLOR), img2 = imread(filenames.back(), IMREAD_COLOR);
      if(!img1.data || !img2.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return 1;
      }

      const int NPOINTS = imgpts1.size();
      if (args.resize_factor > 1) 
      {
         resize(img1, img1, Size(img1.cols / args.resize_factor, 
                  img1.rows / args.resize_factor)); // make smaller for performance and displayablity
         resize(img2, img2, Size(img2.cols / args.resize_factor,
                  img2.rows / args.resize_factor));
         // scale matrix down according to changed resolution
         camera_matrix = camera_matrix / args.resize_factor;
         camera_matrix.at<double>(2,2) = 1;
         for (int i = 0; i < NPOINTS; i++) 
         {
            imgpts1[i] = Point2f(imgpts1[i].x / args.resize_factor, imgpts1[i].y / args.resize_factor);
            imgpts2[i] = Point2f(imgpts2[i].x / args.resize_factor, imgpts2[i].y / args.resize_factor);
         }
      }
      vector<DMatch> matches(NPOINTS);
      vector<KeyPoint> KeyPoints_1(NPOINTS), KeyPoints_2(NPOINTS);
      for (int i = 0; i < NPOINTS; i++) 
      {
         matches[i] = DMatch(i,i,0);
         KeyPoints_1[i] = KeyPoint(imgpts1[i], 10);
         KeyPoints_2[i] = KeyPoint(imgpts2[i], 10);
      }
      if (args.undistort) 
      {
         undistortPoints(imgpts1, imgpts1, camera_matrix, dist_coefficients, noArray(), camera_matrix);
         undistortPoints(imgpts2, imgpts2, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      } 

      double focal = camera_matrix.at<double>(0,0);
      Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

      Mat R, t;
      Mat mask; // inlier mask
      Mat E = findEssentialMat(imgpts1, imgpts2, focal, principalPoint, RANSAC, 0.9999, 5, mask);
      int inliers = recoverPose(E, imgpts1, imgpts2, R, t, focal, principalPoint, mask);
      cout << "Matches used for pose recovery: " << inliers << " of " << imgpts1.size() << endl;

      Mat mtxR, mtxQ;
      Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
      cout << "Translation: " << t.t() << endl;
      cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

      if (args.epilines)
      {
         drawEpilines(Mat(imgpts1), 1, E, img2);
         drawEpilines(Mat(imgpts2), 2, E, img1);
      }

      if (args.draw_matches) 
      {
         Mat img_matches;
         drawMatches(img1, KeyPoints_1, img2, KeyPoints_2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), mask);
         namedWindow("Matches", CV_WINDOW_NORMAL);
         imshow("Matches", img_matches);
         waitKey(0);
      }

      Mat pnts4D;
      Mat P1 = camera_matrix * Mat::eye(3, 4, CV_64FC1), P2;
      Mat p2[2] = { R, t }; 
      hconcat(p2, 2, P2);
      P2 = camera_matrix * P2;

      triangulatePoints(P1, P2, imgpts1, imgpts2, pnts4D);
      pnts4D = pnts4D.t();
      Mat dehomogenized;
      convertPointsFromHomogeneous(pnts4D, dehomogenized);
      dehomogenized = dehomogenized.reshape(1); // instead of 3 channels and 1 col, we want 1 channel and 3 cols
      double mDist = 0;
      int n = 0;
      int pos = 0, neg = 0;
      Mat_<double> row;
      for (int i = 0; i < dehomogenized.rows; i++) 
      {
         row = dehomogenized.row(i);
         double d = row(2);
         if (d > 0) 
         {
            pos++;
            mDist += norm(row);
            n++;
         } else neg++;
      }

      mDist /= n;
      cout << "Mean distance of " << n << " points to camera: " << mDist << " (dehomogenized)" << endl;
      cout << "pos=" << pos << ", neg=" << neg << endl;
   }
   return 0;
}
