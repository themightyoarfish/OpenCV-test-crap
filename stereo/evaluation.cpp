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

tuple<Mat,Mat,double> compute(Mat img1, Mat img2, vector<Point2f> imgpts1, vector<Point2f> imgpts2, int resize_factor = 1, bool epilines = false, bool draw_matches = false)
{
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

   const int NPOINTS = imgpts1.size();
   resize(img1, img1, Size(img1.cols / resize_factor, img1.rows / resize_factor)); 
   resize(img2, img2, Size(img2.cols / resize_factor, img2.rows / resize_factor));
   camera_matrix = camera_matrix / resize_factor;
   camera_matrix.at<double>(2,2) = 1;
   for (int i = 0; i < NPOINTS; i++) 
   {
      imgpts1[i] = Point2f(imgpts1[i].x / resize_factor, imgpts1[i].y / resize_factor);
      imgpts2[i] = Point2f(imgpts2[i].x / resize_factor, imgpts2[i].y / resize_factor);
   }
   vector<DMatch> matches(NPOINTS);
   vector<KeyPoint> KeyPoints_1(NPOINTS), KeyPoints_2(NPOINTS);
   for (int i = 0; i < NPOINTS; i++) 
   {
      matches[i] = DMatch(i,i,0);
      KeyPoints_1[i] = KeyPoint(imgpts1[i], 10);
      KeyPoints_2[i] = KeyPoint(imgpts2[i], 10);
   }
      undistortPoints(imgpts1, imgpts1, camera_matrix, dist_coefficients, noArray(), camera_matrix);
      undistortPoints(imgpts2, imgpts2, camera_matrix, dist_coefficients, noArray(), camera_matrix);

   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat R, t;
   Mat mask; // inlier mask
   Mat E = findEssentialMat(imgpts1, imgpts2, focal, principalPoint, LMEDS);
   Mat F = camera_matrix.t().inv() * E * camera_matrix.inv();
   int inliers = recoverPose(E, imgpts1, imgpts2, R, t, focal, principalPoint, mask);
   cout << "Matches used for pose recovery: " << inliers << " of " << imgpts1.size() << endl;

   Mat mtxR, mtxQ;
   Vec3d angles = RQDecomp3x3(R, mtxR, mtxQ);
   cout << "Translation: " << t.t() << endl;
   cout << "Euler angles [x y z] in degrees: " << angles.t() << endl;

   if (epilines)
   {
      drawEpilines(Mat(imgpts1), 1, F, img2);
      drawEpilines(Mat(imgpts2), 2, F, img1);
   }

   if (draw_matches) 
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
      // check if P2 == R * P1 + t
      /* Mat p2 = camera_matrix * (R * row.t() + t); */
      /* double div = p2.at<double>(2); */
      /* cout << "Reprojected point: "<<p2/div << endl; */
      /* cout << "Actual point: "<<imgpts2[i] << endl; */
      // yep, that's right. I also checked if p1 = camera_matrix * P1, so the
      // triangulated points are indeed in the left cam's coordinate system
      double d = row(2);
      if (d > 0) 
      {
         pos++;
         mDist += norm(row);
         n++;
      } else neg++;
   }

   mDist /= n;
   /* cout << "Mean distance of " << n << " points to camera: " << mDist << " (dehomogenized)" << endl; */
   /* cout << "pos=" << pos << ", neg=" << neg << endl; */
   return make_tuple(R,t,mDist);
}

string pathForFiles(string img1, string img2)
{
      char filename[100];

      GET_BASE_NAME(img1);
      GET_BASE_NAME(img2);
      sprintf(filename, "../Data/Bahnhof/imgpts_%s->%s.txt",img1.c_str(),img2.c_str());
      return string(filename);
}
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);

   Mat firstFrame, secondFrame, reference, currentFrame;
   Mat tFirstRef, RFirstRef, tFirstCurrent, RFirstCurrent;
   double goalScale;

   vector<Point2f> imgpts1, imgpts2; // reusabe vectors for manually labeled points
   firstFrame = imread(filenames.back());
   secondFrame = imread(filenames[4]); // 5.JPG
   reference = imread(filenames.front());

   tie(imgpts2,imgpts1) = readPtsFromFile(pathForFiles(filenames.front(),filenames.back())); // swap vectors since first frame points come at the end

   tie(RFirstRef, tFirstRef, ignore) = compute(firstFrame, reference, imgpts1, imgpts2, args.resize_factor, args.epilines, args.draw_matches);

   tie(imgpts2,imgpts1) = readPtsFromFile(pathForFiles(filenames[1],filenames.back())); // swap vectors since first frame points come at the end
   tie(ignore, ignore, goalScale) = compute(firstFrame, secondFrame, imgpts1, imgpts2, args.resize_factor);

   cout << "<<<<<< Preprocessing done." << endl;

   for (int i = 1; i < filenames.size() -1; i++) 
   {
      string filename = pathForFiles(filenames[i],filenames.back());
      cout << "Reading from file " << filename << endl;
      tie(imgpts2, imgpts1) = readPtsFromFile(filename); // swap since first frame pts come at the bottom

      Mat img2 = imread(filenames[i], IMREAD_COLOR), img1 = firstFrame; // make first frame left image
      if(!img1.data || !img2.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return 1;
      }

      double world_scale;
      tie(RFirstCurrent,tFirstCurrent,world_scale) = compute(img1, img2, imgpts1, imgpts2, args.resize_factor, args.epilines, args.draw_matches);
      PRINT("world scale:",world_scale);

      Mat RCurrentRef = RFirstRef * RFirstCurrent.t();
      Mat tCurrentRef = -(RFirstRef * RFirstCurrent.t() * tFirstCurrent + tFirstRef);
      Mat mtxR, mtxQ;
      Vec3d angles = RQDecomp3x3(RCurrentRef, mtxR, mtxQ);
      cout << "============\n";
      PRINT("Deduced euler angles [x,y,z]:", angles.t());
      PRINT("Translation: ", tCurrentRef / norm(tCurrentRef) * (world_scale / goalScale));
      cout << "============\n" << endl;

   }
   return 0;
}
