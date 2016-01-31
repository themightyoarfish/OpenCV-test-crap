#ifndef SERIALIZATION_H
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
using cv::Point2f;
using std::ifstream;
ifstream& operator>>(ifstream& s, Point2f& val)
{
   s.seekg(1, std::ios_base::cur);
   s >> val.x;
   s.seekg(1, std::ios_base::cur);
   s >> val.y;
   s.seekg(1, std::ios_base::cur);
   return s;
}

template<typename T1, typename T2> void serialize_vector(const std::vector<std::pair<T1,T2>>& vec, std::string filename);
template<typename T1, typename T2> std::vector<std::pair<T1,T2>> deserialize_vector(std::string filename);
#include "serialization.tpp"
#define SERIALIZATION_H
#endif /* end of include guard: SERIALIZATION_H */
