#ifndef SERIALIZATION_H
#include <vector>
template<typename T1, typename T2> void serialize_vector(const std::vector<std::pair<T1,T2>>& vec, std::string filename);
template<typename T1, typename T2> std::vector<std::pair<T1,T2>> deserialize_vector(std::string filename);
#include "serialization.tpp"
#define SERIALIZATION_H
#endif /* end of include guard: SERIALIZATION_H */
