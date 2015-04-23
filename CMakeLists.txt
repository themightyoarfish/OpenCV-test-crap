cmake_minimum_required(VERSION 2.8)

set( CMAKE_CXX_FLAGS         "-O0 -g -w" )
find_package( OpenCV REQUIRED )
add_executable( canny canny.cpp )
add_executable( transpTest transpTest.cpp)
target_link_libraries( canny ${OpenCV_LIBS} )
target_link_libraries( transpTest ${OpenCV_LIBS} )
