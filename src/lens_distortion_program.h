#pragma once

#include "ami_image/image.h"

#include <string>
#include <vector>

namespace lens_distortion {

const int default_max_lines = 100; //maximum number of lines estimated
const float default_angle_resolution = 0.1; // angle discretization step (in degrees)
const float default_distance_resolution = 1.; // line distance discretization step
const float default_distortion_parameter_resolution = 0.1; //distortion parameter discretization step

/// \brief This class holds the result of the lens distortion algorithm
class UndistortionResult{
  public:
    bool success = false;

    // Algorithm parameters and results
    std::string tmodel;
    bool opt_c = false;
    double k1 = 0.0;
    double k2 = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    // Undistorted image itself
    ami::image<unsigned char> undistorted;
    int width = 0;
    int height = 0;

    // Method to get the undistorted image as a numpy compatible array of unsigned char
    std::vector<unsigned char> getUndistortedAsArray() const;
};

ami::image<unsigned char> imageFromArray(
  const std::vector<unsigned char>& data_array, 
  const int width, 
  const int height,
  const int channels
);


int processFile(
  UndistortionResult & undistortion_result,
  const std::string & input_filepath,
  const std::string& output_folder,
  const int width,
  const int height,
  const float canny_high_threshold,
  const float initial_distortion_parameter,
  const float final_distortion_parameter,
  const float distance_point_line_max_hough,
  const float angle_point_orientation_max_difference,
  const std::string& tmodel,
  const std::string& s_opt_c,
  const bool write_intermediates,
  const bool write_output,
  const int max_lines,
  const float angle_resolution,
  const float distance_resolution,
  const float distortion_parameter_resolution
);

/// \brief This runs the algorithm, the main function will be a wrapper for this one
int runAlgorithm(
  const std::vector<std::string>& input_files,
  const std::string& output_folder,
  const int width,
  const int height,
  const float canny_high_threshold,
  const float initial_distortion_parameter,
  const float final_distortion_parameter,
  const float distance_point_line_max_hough,
  const float angle_point_orientation_max_difference,
  const std::string& tmodel,
  const std::string& s_opt_c,
  const bool write_intermediates,
  const bool write_output
);

}