set(CMAKE_CXX_COMPILER "/opt/local/bin/g++")
set(CMAKE_C_COMPILER "/opt/local/bin/gcc")
set( CMAKE_CXX_FLAGS         "-O3 -w -lstdc++" ) # -DSHOW_IMGS to display images
find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   set(OpenCV_LIB_DIR "/usr/local/lib/")
endif(OpenCV_FOUND)

get_cmake_property(_v VARIABLES)
foreach(_v ${_v})
   message(STATUS "${_v}=${${_v}}")
endforeach()



add_executable( cannyDetector canny/canny.cpp )
add_executable( transpTest edgesToPng/transpTest.cpp)
add_executable( correctImage calibration/correctImage.cpp)
add_executable( stereo stereo/stereo.cpp)

target_link_libraries( cannyDetector ${OpenCV_LIBS} )
target_link_libraries( transpTest ${OpenCV_LIBS} )
target_link_libraries( correctImage ${OpenCV_LIBS} )
target_link_libraries( stereo ${OpenCV_LIBS} )
