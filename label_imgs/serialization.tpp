#include <fstream>
#include <string>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::make_pair;
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
            cout << "Writing " << iter->first << " and " << iter->second << endl;
            out_file << iter->first << iter->second;
         }
      }
   } else cerr << "Could not open file " << filename << endl;
}
template<typename T1, typename T2> vector<pair<T1,T2>> deserialize_vector(string filename)
{
   ifstream in_file(filename);
   vector<pair<T1,T2>> vec;
   if (in_file.is_open() && in_file.good())
   {
      size_t size;
      in_file >> size;
      vec.resize(size);
      for (int i = 0; i < size; i++) 
      {
         if (!in_file.good()) 
         {
            cerr << "Could not read from file " << filename << endl;
            break;
         } else
         {
            T1 t1; T2 t2;
            in_file >> t1;
            in_file >> t2;
            vec[i] = make_pair(t1,t2);
         }
      }
   } else cerr << "Could not open file " << filename << endl;
   return vec;
}
