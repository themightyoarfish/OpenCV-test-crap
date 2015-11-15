set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w") 
set(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wreturn-type")
find_package(OpenCV REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH) # manual specification of the OpenCVConfig.cmake path is necessary since otherwise, cmake assigns /opt/local a higher priority and ocv2.4 will be used

find_package(Boost REQUIRED filesystem system)

if(OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
   include_directories(".")
   include_directories("stereo")
   include_directories("calibration")
   include_directories("triangulate")
   include_directories("Data/Series/Set6\ labeled")
   include_directories("label_imgs")

   add_executable(cannyDetector canny/canny.cpp)
   add_executable(transpTest edgesToPng/transpTest.cpp)
   add_executable(correctImage calibration/correctImage.cpp)
   add_executable(stereo stereo/stereo_v3_main.cpp stereo/stereo_v3.cpp triangulate/Triangulation.cpp)
   add_executable(triangulate triangulate/triangulate.cpp stereo/stereo_v3.cpp triangulate/Triangulation.cpp)
   add_executable(scale_degeneration_test stereo/scale_degeneration_test.cpp stereo/stereo_v3.cpp triangulate/Triangulation.cpp)
   add_executable(manually_labelled stereo/manually_labeled.cpp stereo/stereo_v3.cpp triangulate/Triangulation.cpp)
   add_executable(castle_data stereo/castle_data.cpp stereo/stereo_v3.cpp)
   add_executable(evaluation stereo/evaluation.cpp stereo/stereo_v3.cpp)
   add_executable(solvePnP solvePnP/solvePnP.cpp solvePnP/ImageSeries.cpp calibration/CalibrationFileReader.cpp)

   target_link_libraries(cannyDetector ${OpenCV_LIBS})
   target_link_libraries(transpTest ${OpenCV_LIBS})
   target_link_libraries(correctImage ${OpenCV_LIBS})
   target_link_libraries(stereo ${OpenCV_LIBS})
   target_link_libraries(triangulate ${OpenCV_LIBS})
   target_link_libraries(scale_degeneration_test ${OpenCV_LIBS})
   target_link_libraries(manually_labelled ${OpenCV_LIBS})
   target_link_libraries(castle_data ${OpenCV_LIBS})
   target_link_libraries(evaluation ${OpenCV_LIBS})


   target_link_libraries(solvePnP ${OpenCV_LIBS})

   if(Boost_FOUND)
      include_directories(${Boost_INCLUDE_DIR})
      add_executable(label_imgs label_imgs/main.cpp
         label_imgs/DualImageWindow.cpp)
      target_link_libraries(label_imgs ${Boost_LIBRARIES} ${OpenCV_LIBS})
   endif(Boost_FOUND)
endif(OpenCV_FOUND)

#get_cmake_property(_v VARIABLES)
#foreach(_v ${_v})
#   message(STATUS "${_v}=${${_v}}")
#endforeach()
