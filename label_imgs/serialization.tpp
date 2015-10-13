#include <fstream>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;
using std::ofstream;
using std::vector;
using std::pair;
using std::ios;
using std::string;

template<typename T1, typename T2> void serialize_vector(const vector<pair<T1,T2>>& vec, string filename)
{
   ofstream out_file(filename, ios::binary);
   if (out_file.is_open() && out_file.good())
   {
      out_file << vec.size();
      for (auto iter = vec.begin() ; iter != vec.end() ; iter++) 
      {
         if (!out_file.good()) 
         {
            cerr << "Could not write to file " << filename << endl;
            break;
         } else
         {
            out_file << iter->first << iter->second;
         }
      }
   }
}
template<typename T1, typename T2> vector<pair<T1,T2>> deserialize_vector(string filename)
{
}
