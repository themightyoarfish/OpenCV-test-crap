/**
 * @file CalibrationFileReader.cpp
 * @brief Implementation of @c CalibrationFileReader class
 * @date 09/06/15
 * @author Rasmus Diederichsen
 */
#include "CalibrationFileReader.h"
#include <iostream>

using namespace cv;
using namespace std;

Mat CalibrationFileReader::getCameraMatrix() const
{
    return mCalMat;
}

Mat CalibrationFileReader::getDistortionCoeffs() const
{
    return mDistCoeffs;
}

CalibrationFileReader::CalibrationFileReader(string& fileName)
{
    mFileName = fileName;
    FileStorage fs(fileName, FileStorage::READ);
    if (fs.isOpened())
    {
        fs[CAM_MAT_KEY] >> mCalMat;
        fs[DIST_COEFF_KEY] >> mDistCoeffs;
        fs.release();
    } else cerr << "Failed to read calibration data \"" << fileName << "\"" << endl;

}
