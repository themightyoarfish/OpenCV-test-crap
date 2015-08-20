#include "stereo_v3.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

#define BAHNHOF
#define FEATURES

#ifdef BAHNHOF
static string TRUTH = "../Data/Bahnhof/TRUTH.txt";
#else
static string TRUTH = "../Data/Gut Rosenkrantz/TRUTH.txt";
#endif

static vector<string> bahnhof = {
   "../Data/Bahnhof/ref_corrected.JPG",
   "../Data/Bahnhof/2.JPG",
   "../Data/Bahnhof/3.JPG",
   "../Data/Bahnhof/4.JPG",
   "../Data/Bahnhof/5.JPG",
   "../Data/Bahnhof/first_frame_centered.JPG",
};

static vector<string> gut_rosenkrantz = {
   "../Data/Gut Rosenkrantz/0_ref.JPG",
   "../Data/Gut Rosenkrantz/1.JPG",
   "../Data/Gut Rosenkrantz/2.JPG",
   "../Data/Gut Rosenkrantz/3.JPG",
   "../Data/Gut Rosenkrantz/4.JPG",
   "../Data/Gut Rosenkrantz/5.JPG",
   /* "../Data/Gut Rosenkrantz/8.JPG", */ // this is from the same spot as the first frame
   "../Data/Gut Rosenkrantz/9.JPG",
   "../Data/Gut Rosenkrantz/first_frame_centered.JPG",
};

static tuple<Mat,Mat,double> realTransLationForImage(string image_base_name)
{
   ifstream file(TRUTH, ios::in);
   if (!file.is_open()) 
   {
      cerr << "Could not read truth data." << endl;
      return make_tuple(Mat(),Mat(),0);
   }
   char line[200];
   size_t s = image_base_name.length();
   double theta_x,theta_y,theta_z,trans_x,trans_y,trans_z,dist_ratio;
   bool found = false;
   char unused[1024];
   while (!file.eof() && !found)
   {
      file.getline(line, 199);
      if (file.eof()) break;
      if (line[0] == '"' || line[0] == '\0') continue;
      if (string(line,s) == image_base_name) 
      {
         found = true;
         sscanf(line, "%s [%lf,%lf,%lf] [%lf,%lf,%lf] %lf",unused, &theta_x,&theta_y,&theta_z,&trans_x,&trans_y,&trans_z,&dist_ratio);
         break;
      }
   }
   if (found) 
   {
      Mat_<double> rot(3,1), trans(3,1);
      rot << theta_x, theta_y, theta_z;
      trans << trans_x, trans_y, trans_z;
      return make_tuple(rot, trans, dist_ratio);
   }
      return make_tuple(Mat(),Mat(),0);
}


tuple<vector<Point2f>, vector<Point2f>, double> readPtsFromFile(string filename)
{
   vector<Point2f> imgpts1, imgpts2;
   ifstream file(filename, ios::in);
   if (!file.is_open()) 
   {
      cerr << "Wtf, dude." << endl;
      return make_tuple(vector<Point2f>(), vector<Point2f>(), -1.0);
   }
   char line[100];
   bool first_part = true;
   double distance;
   while (!file.eof()) 
   {
      file.getline(line, 30);
      if (file.eof()) break;
      if (line[0] == '"') 
      {
         if (line[1] == '[') 
         {
            const char* format = "\"[x,y] = [%lf,%lf]\"";
            double x, y;
            sscanf(line, format, &x, &y);
            distance = sqrt(x * x + y * y);
         } 
         continue;
      }
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
   return make_tuple(imgpts1, imgpts2, distance);
}

#define GET_BASE_NAME(file)\
{\
   size_t last_backslash = file.find_last_of("/");\
   size_t ext_start = file.find_last_of(".");\
   file = file.substr(last_backslash + 1, ext_start - last_backslash - 1);\
}

tuple<Mat,Mat,double> compute(Mat img1, Mat img2, vector<Point2f> imgpts1, vector<Point2f> imgpts2, int resize_factor = 1, bool epilines = false, bool draw_matches = false, bool points_are_resized = false)
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
   if (!points_are_resized)
   {
      for (int i = 0; i < NPOINTS; i++) 
      {
         imgpts1[i] = Point2f(imgpts1[i].x / resize_factor, imgpts1[i].y / resize_factor);
         imgpts2[i] = Point2f(imgpts2[i].x / resize_factor, imgpts2[i].y / resize_factor);
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
   undistortPoints(imgpts1, imgpts1, camera_matrix, dist_coefficients, noArray(), camera_matrix);
   undistortPoints(imgpts2, imgpts2, camera_matrix, dist_coefficients, noArray(), camera_matrix);

   double focal = camera_matrix.at<double>(0,0);
   Point2d principalPoint(camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));

   Mat R, t;
   Mat mask; // inlier mask
   Mat E = findEssentialMat(imgpts1, imgpts2, focal, principalPoint, RANSAC, 0.999, 3, mask);
   Mat F = camera_matrix.t().inv() * E * camera_matrix.inv();
   correctMatches(F, imgpts1, imgpts2, imgpts1, imgpts2);
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

tuple<vector<Point2f>, vector<Point2f>> getFeatureMatches(Mat img1, Mat img2, CommandArgs args)
{
   if (args.resize_factor > 1) 
   {
      resize(img1, img1, Size(img1.cols / args.resize_factor, img1.rows / args.resize_factor)); 
      resize(img2, img2, Size(img2.cols / args.resize_factor, img2.rows / args.resize_factor));
   }
   vector<KeyPoint> KeyPoints_1, KeyPoints_2;
   Mat descriptors_1, descriptors_2;
   Ptr<Feature2D> feat_detector;
   if (args.detector == DETECTOR_KAZE) 
   {
      feat_detector = AKAZE::create(args.detector_data.upright ? AKAZE::DESCRIPTOR_MLDB_UPRIGHT : AKAZE::DESCRIPTOR_MLDB, 
            args.detector_data.descriptor_size,
            args.detector_data.descriptor_channels,
            args.detector_data.threshold,
            args.detector_data.nOctaves,
            args.detector_data.nOctaveLayersAkaze);

   } else 
      feat_detector = xfeatures2d::SURF::create(args.detector_data.minHessian, 
            args.detector_data.nOctaves, args.detector_data.nOctaveLayersAkaze, args.detector_data.extended, args.detector_data.upright);

   feat_detector->detectAndCompute(img1, noArray(), KeyPoints_1, descriptors_1);
   feat_detector->detectAndCompute(img2, noArray(), KeyPoints_2, descriptors_2);

   cout << "Number of feature points (img1, img2): " << "(" << KeyPoints_1.size() << ", " << KeyPoints_2.size() << ")" << endl;

   // Find correspondences
   BFMatcher matcher;
   vector<DMatch> matches;
   if (args.use_ratio_test) 
   {
      if (args.detector == DETECTOR_KAZE) 
         matcher = BFMatcher(NORM_HAMMING, false);
      else matcher = BFMatcher(NORM_L2, false);

      vector<vector<DMatch>> match_candidates;
      const float ratio = args.ratio;
      matcher.knnMatch(descriptors_1, descriptors_2, match_candidates, 2);
      for (int i = 0; i < match_candidates.size(); i++)
         if (match_candidates[i][0].distance < ratio * match_candidates[i][1].distance)
            matches.push_back(match_candidates[i][0]);

      cout << "Number of matches passing ratio test: " << matches.size() << endl;

   } else
   {
      if (args.detector == DETECTOR_KAZE) 
         matcher = BFMatcher(NORM_HAMMING, true);
      else matcher = BFMatcher(NORM_L2, true);
      matcher.match(descriptors_1, descriptors_2, matches);
      cout << "Number of matching feature points: " << matches.size() << endl;
   }


   // Convert correspondences to vectors
   vector<Point2f>imgpts1,imgpts2;

   for(unsigned int i = 0; i < matches.size(); i++) 
   {
      imgpts1.push_back(KeyPoints_1[matches[i].queryIdx].pt); 
      imgpts2.push_back(KeyPoints_2[matches[i].trainIdx].pt); 
   }

   return make_tuple(imgpts1, imgpts2);
}

string pathForFiles(string img1, string img2)
{
   char filename[100];

   GET_BASE_NAME(img1);
   GET_BASE_NAME(img2);
#ifdef BAHNHOF
   sprintf(filename, "../Data/Bahnhof/imgpts_%s->%s.txt",img1.c_str(),img2.c_str());
#else
   sprintf(filename, "../Data/Gut Rosenkrantz/imgpts_%s->%s.txt",img1.c_str(),img2.c_str());
#endif
   return string(filename);
}
int main(int argc, char *argv[])
{
   CommandArgs args = parse_args(argc, argv);

#ifdef BAHNHOF
   vector<string> filenames = bahnhof;
#else
   vector<string> filenames = gut_rosenkrantz;
#endif

   Mat firstFrame, secondFrame, reference, currentFrame;
   Mat tFirstRef, RFirstRef, tFirstCurrent, RFirstCurrent;
   double goalScale;
   double dist_first_ref, dist_first_second;

#ifdef BAHNHOF
   string secondFrameName = filenames[4]; // 5.JPG
#else
   string secondFrameName = filenames[1]; // 1.JPG
#endif

   vector<Point2f> imgpts1, imgpts2; // reusabe vectors for manually labeled points
   firstFrame = imread(filenames.back());
   secondFrame = imread(secondFrameName); 
   reference = imread(filenames.front());

   bool points_are_resized;
#ifndef FEATURES
   tie(imgpts2,imgpts1, dist_first_ref) = readPtsFromFile(pathForFiles(filenames.front(),filenames.back())); // swap vectors since first frame points come at the end
   points_are_resized = false;
#else
   tie(ignore, ignore, dist_first_ref) = readPtsFromFile(pathForFiles(filenames.front(),filenames.back())); // get only distance (sloppy)
   tie(imgpts1,imgpts2) = getFeatureMatches(firstFrame,reference,args); 
   points_are_resized = true;
#endif

   tie(RFirstRef, tFirstRef, ignore) = compute(firstFrame, reference, imgpts1,
         imgpts2, args.resize_factor, args.epilines, args.draw_matches,
         points_are_resized);

   // compute scale with first and second frames
#ifndef FEATURES
   tie(imgpts2,imgpts1,dist_first_ref) =
      readPtsFromFile(pathForFiles(filenames.front(),filenames.back())); // swap vectors since first frame points come at the end
   points_are_resized = false;
#else
   tie(ignore,ignore,dist_first_ref) =
      readPtsFromFile(pathForFiles(filenames.front(),filenames.back())); // get only distance
   tie(imgpts1,imgpts2) = getFeatureMatches(firstFrame,reference,args); 
   points_are_resized = true;
#endif
   tie(ignore, ignore, goalScale) = compute(firstFrame, reference, imgpts1,
         imgpts2, args.resize_factor, args.epilines, args.draw_matches,
         points_are_resized);
   cout << "Goal scale: " << goalScale << endl;

   cout << "<<<<<< Preprocessing done. >>>>>>" << endl;

   /* Open data tile with format fname x y z thetax thetay theta z dist_ratio realx realy realz realtthetax realthetay tealthetaz realdist_ratio */
   vector<string> header = {
      "fname", "x", "y", "z", "thetax", "thetay", "thetaz", "dist_ratio",
      "realx", "realy", "realz", "realthetax", "realthetay", "realthetaz",
      "realdist_ratio"
   };
   char datafilename[200];
   switch(args.detector)
   {
      case DETECTOR_SIFT:
#ifdef BAHNHOF
         sprintf(datafilename, "bahnhof_detector_%s_resize_%d_ratio_%lf.dat", "SIFT", args.resize_factor, args.ratio);
#else
         sprintf(datafilename, "gut_rosenkrantz_detector_%s_resize_%d_ratio_%lf.dat", "SIFT", args.resize_factor, args.ratio);
#endif
         break;
      case DETECTOR_KAZE:
#ifdef BAHNHOF
         sprintf(datafilename, "bahnhof_detector_%s_resize_%d_ratio_%lf.dat", "KAZE", args.resize_factor, args.ratio);
#else
         sprintf(datafilename, "gut_rosenkrantz_detector_%s_resize_%d_ratio_%lf.dat", "KAZE", args.resize_factor, args.ratio);
#endif
         break;
   }
   ofstream datafile(datafilename);
   if (datafile.is_open()) 
   {
      for (vector<string>::iterator i = header.begin(); i != header.end(); i++) 
         datafile << setw(16) << *i;
   }
   datafile << endl;

   for (int i = 0; i < filenames.size() -1; i++) 
   {
      string filename = pathForFiles(filenames[i],filenames.back());
#ifndef FEATURES
      cout << "Reading from file " << filename << endl;
#endif

      Mat img2 = imread(filenames[i], IMREAD_COLOR), img1 = firstFrame; // make first frame left image
      if(!img1.data || !img2.data) 
      {
         cout << "At least one of the images has no data." << endl;
         return 1;
      }

      double camera_distance;
#ifndef FEATURES
      tie(imgpts2, imgpts1, camera_distance) = readPtsFromFile(filename); // swap since first frame pts come at the bottom
#else
      tie(ignore, ignore, camera_distance) = readPtsFromFile(filename); // only get distance
      tie(imgpts1,imgpts2) = getFeatureMatches(img1,img2,args); 
      points_are_resized = true;
#endif

      double world_scale;
      tie(RFirstCurrent,tFirstCurrent,world_scale) = compute(img1, img2, imgpts1, imgpts2, args.resize_factor, args.epilines, args.draw_matches, points_are_resized);
      double ratio = world_scale / goalScale;

      Mat RCurrentRef = RFirstRef * RFirstCurrent.t();
      Mat tCurrentRef = -RFirstRef * RFirstCurrent.t() * tFirstCurrent + tFirstRef;
      tCurrentRef = tCurrentRef / norm(tCurrentRef);
      Mat mtxR, mtxQ;
      cout << "Translation to reference: " << tCurrentRef.t() << endl;
      Vec3d angles = RQDecomp3x3(RCurrentRef, mtxR, mtxQ);
      cout << "Rotation to reference: " << angles << endl;
      cout << "Ratio: " << ratio << endl;

      Mat realRot, realTrans;
      double realDistRatio;
      string fname = filenames[i];
      GET_BASE_NAME(fname);
      tie(realRot,realTrans,realDistRatio) = realTransLationForImage(fname);

      cout << "================================================\n";
      /* PRINT("Deduced euler angles [x,y,z]:", angles.t()); */
      /* PRINT("Translation: ", tCurrentRef / norm(tCurrentRef) * ratio); */
      /* PRINT("world scale ratio:",world_scale/goalScale); */
      /* PRINT("Real ratio:",1.0/(camera_distance/dist_first_ref)); */
      cout << "Translation difference: " << (tCurrentRef - realTrans).t() << endl;
      cout << "Rotation difference: " << (Mat(angles) - realRot).t() << endl;
      cout << "Ratio difference: " << ratio - realDistRatio << endl;
      cout << "================================================\n" << endl;

   /* fname x y z thetax thetay thetaz dist_ratio realx realy realz realtthetax realthetay tealthetaz realdist_ratio */
      string base_file_name = filenames[i];
      GET_BASE_NAME(base_file_name);
      datafile << setw(16) << base_file_name;
      datafile << setw(16) << tCurrentRef.at<double>(0);
      datafile << setw(16) << tCurrentRef.at<double>(1);
      datafile << setw(16) << tCurrentRef.at<double>(2);
      datafile << setw(16) << angles[0];
      datafile << setw(16) << angles[1];
      datafile << setw(16) << angles[2];
      datafile << setw(16) << ratio;
      datafile << setw(16) << realTrans.at<double>(0);
      datafile << setw(16) << realTrans.at<double>(1);
      datafile << setw(16) << realTrans.at<double>(2);
      datafile << setw(16) << realRot.at<double>(0);
      datafile << setw(16) << realRot.at<double>(1);
      datafile << setw(16) << realRot.at<double>(2);
      datafile << setw(16) << realDistRatio;
      datafile << endl;

   }
   return 0;
}
