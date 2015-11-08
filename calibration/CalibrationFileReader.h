/**
 * @file CalibrationFileReader.h
 * @date 09/06/15
 * @author Rasmus Diederichsen
 * @brief Declaration of @c CalibrationFileReader class
 *
 */
#ifndef __refoto_app__CalibrationFileReader__
#define __refoto_app__CalibrationFileReader__

#include <string>
#include <opencv2/opencv.hpp>

/*! @brief The key used for retrieving camera matrix from the OpenCV calibration file */
const std::string CAM_MAT_KEY("Camera_Matrix");
/*! @brief The key used for retrieving distortion matrix from the OpenCV calibration file */
const std::string DIST_COEFF_KEY("Distortion_Coefficients");

/**
 * @class CalibrationFileReader
 *
 * On iOS, use it like this:
  @code
  NSString* fileName = @"ipad_camera_params.xml"; // or whatever
  std::string fName([[[[NSBundle mainBundle] resourcePath]
                      stringByAppendingPathComponent:fileName]  cStringUsingEncoding:NSMacOSRomanStringEncoding]);
  CalibrationFileReader c(fname);
  @encode
 */
class CalibrationFileReader
{
    std::string mFileName; ///< Name of the opencv xml file containing calibration params
    cv::Mat mCalMat; ///< camera matrix parsed from the file
    cv::Mat mDistCoeffs; ///< distortion coefficients parsed from the file
    
public:
    /**
     * @brief Initialise a new @c CalibrationFileReader
     * @param fileName Path pointing to calibration data written with OpenCV's @c FileStorage class. The keys used for writing camera and distortion matrices must be CAM_MAT_KEY and DIST_COEFF_KEY.
     */
    CalibrationFileReader(std::string& fileName);

    /**
     * @brief Get the camera matrix
     * @return A @c cv::Mat object of 3x3 dimension containing camera parameters as usual
     */
    cv::Mat getCameraMatrix() const;

    /**
     * @brief Get the distortion coefficients.
     * @return A @c cv::Mat object of [3-5]x1 dimension containing distortion coefficients
     */
    cv::Mat getDistortionCoeffs() const;
};

#endif /* defined(__refoto_app__CalibrationFileReader__) */
