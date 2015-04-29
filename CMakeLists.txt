set( CMAKE_CXX_FLAGS         "-O3 -w" ) # -DSHOW_IMGS to display images
find_package( OpenCV REQUIRED )

add_executable( cannyDetector canny/canny.cpp )
add_executable( transpTest edgesToPng/transpTest.cpp)
add_executable( correctImage calibration/correctImage.cpp)
add_executable( stereo stereo/stereo.cpp)

target_link_libraries( cannyDetector ${OpenCV_LIBS} )
target_link_libraries( transpTest ${OpenCV_LIBS} )
target_link_libraries( correctImage ${OpenCV_LIBS} )
target_link_libraries( stereo ${OpenCV_LIBS} )
