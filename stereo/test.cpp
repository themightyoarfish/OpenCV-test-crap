#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double computeReprojectionError(vector<Point2f>& imgpts1, vector<Point2f>& imgpts2, Mat& inlier_mask, const Mat& F)
{

   double err = 0;
   vector<Vec3f> lines[2];
   int npt = sum(inlier_mask)[0]; 

   // strip outliers so validation is constrained to the correspondences
   // which were used to estimate F
   vector<Point2f> imgpts1_copy(npt), 
      imgpts2_copy(npt);
      static int c = 0;
   for (int k = 0; k < inlier_mask.size().height; k++) 
   {
      if (inlier_mask.at<uchar>(0,k) == 1) 
      {
         imgpts1_copy[c] = imgpts1[k];
         imgpts2_copy[c] = imgpts2[k];
         c++;
      } 
   }

   Mat imgpt[2] = { Mat(imgpts1_copy), Mat(imgpts2_copy) };
   computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);
   computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
   for(int j = 0; j < npt; j++ )
   {
      double errij = fabs(imgpts1_copy[j].x*lines[1][j][0] +
            imgpts1_copy[j].y*lines[1][j][1] + lines[1][j][2]) +
         fabs(imgpts2_copy[j].x*lines[0][j][0] +
               imgpts2_copy[j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
   }
   return err / npt;
}

int main(int argc, const char *argv[])
{
#include "AUTO_pts.h"
   Mat_<uchar> mask = Mat::ones(imgpts1.size(), 1, CV_8UC1);
   Mat_<double> K(3,3);
   K << 2988.063166886338, 0, 1631.5,
     0, 2988.063166886338, 1223.5,
     0, 0, 1;
  
   Mat F = findFundamentalMat(imgpts1, imgpts2, CV_FM_7POINT);
   cout << computeReprojectionError(imgpts1, imgpts2, mask, F) << endl;
   cout << computeReprojectionError(imgpts1, imgpts2, mask, F) << endl;

   return 0;
}
