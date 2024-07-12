/*
This is a pybind of the following:

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
  const std::string& s_opt_c
){

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "../src/lens_distortion_program.h"

/*

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
  const float distortion_parameter_resolution,
  const lens_distortion_model& ini_ldm
)
*/

namespace py = pybind11;

PYBIND11_MODULE(lens_distortion_pybind, m) {
    m.doc() = "Lens distortion correction algorithm"; // optional module docstring

    m.def("runAlgorithm", &lens_distortion::runAlgorithm, "A function that runs the lens distortion correction algorithm",
        py::arg("input_files"),
        py::arg("output_folder"),
        py::arg("width"),
        py::arg("height"),
        py::arg("canny_high_threshold"),
        py::arg("initial_distortion_parameter"),
        py::arg("final_distortion_parameter"),
        py::arg("distance_point_line_max_hough"),
        py::arg("angle_point_orientation_max_difference"),
        py::arg("tmodel"),
        py::arg("s_opt_c"),
        py::arg("write_intermediates") = false,
        py::arg("write_output") = true
    );
    
    py::class_<lens_distortion::UndistortionResult>(m, "UndistortionResult")
        .def(py::init<>())
        .def_readonly("success", &lens_distortion::UndistortionResult::success)
        .def_readonly("tmodel", &lens_distortion::UndistortionResult::tmodel)
        .def_readonly("opt_c", &lens_distortion::UndistortionResult::opt_c)
        .def_readonly("k1", &lens_distortion::UndistortionResult::k1)
        .def_readonly("k2", &lens_distortion::UndistortionResult::k2)
        .def_readonly("cx", &lens_distortion::UndistortionResult::cx)
        .def_readonly("cy", &lens_distortion::UndistortionResult::cy)
        .def_readonly("width", &lens_distortion::UndistortionResult::width)
        .def_readonly("height", &lens_distortion::UndistortionResult::height)
        .def("undistorted", &lens_distortion::UndistortionResult::getUndistortedAsArray);

    m.def("processFile", &lens_distortion::processFile, "A function that processes a single file",
        py::arg("undistortion_result"),
        py::arg("input_filepath"),
        py::arg("output_folder"),
        py::arg("width"),
        py::arg("height"),
        py::arg("canny_high_threshold"),
        py::arg("initial_distortion_parameter"),
        py::arg("final_distortion_parameter"),
        py::arg("distance_point_line_max_hough"),
        py::arg("angle_point_orientation_max_difference"),
        py::arg("tmodel"),
        py::arg("s_opt_c"),
        py::arg("write_intermediates") = false,
        py::arg("write_output") = false,
        py::arg("max_lines") = lens_distortion::default_max_lines,
        py::arg("angle_resolution") = lens_distortion::default_angle_resolution,
        py::arg("distance_resolution") = lens_distortion::default_distance_resolution,
        py::arg("distortion_parameter_resolution") = lens_distortion::default_distortion_parameter_resolution
    );
}
