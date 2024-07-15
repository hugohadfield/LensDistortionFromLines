#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>
#include "../src/ami_lens_distortion/lens_distortion_procedures.h"
#include "../src/lens_distortion_program.h"

namespace py = pybind11;

const double DEFAULT_OPENCV_FOCAL_LENGTH = 1000.0;


/// \brief This function runs the lens distortion correction algorithm.
int undistortDivisionModel(
    lens_distortion::UndistortionResult & result,
    const std::vector<unsigned char> & input_image_bytes,
    double d1,
    double d2,
    double cx,
    double cy,
    int w,
    int h,
    const int image_amplification_factor
){

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    ami::image<unsigned char> input_image = lens_distortion::imageFromArray(input_image_bytes, w, h, 3);    
    lens_distortion_model d_model;
    d_model.set_type(DIVISION);
    std::vector<double> d = {1.0, d1, d2};
    d_model.set_d(d);
    d_model.set_distortion_center({cx, cy});

    std::chrono::steady_clock::time_point after_setp = std::chrono::steady_clock::now();
    std::cout << "Time to unpack and set parameters: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_setp - begin).count() << " ms" << std::endl;

    const ami::image<unsigned char> undistorted = undistort_quotient_image_inverse(
        input_image,
        d_model,
        image_amplification_factor
    );

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time to undistort after params: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - after_setp).count() << " ms" << std::endl;

    result.success = true;
    result.tmodel = "division";
    result.opt_c = false;
    result.k1 = d1;
    result.k2 = d2;
    result.cx = cx;
    result.cy = cy;
    result.width = undistorted.width();
    result.height = undistorted.height();
    result.undistorted.clear();
    result.undistorted = undistorted;
    return 1;
}


double opencv_fisheye_polynomial(
    double r, 
    double k1, 
    double k2, 
    double k3, 
    double k4, 
    double k5, 
    double k6,
    double opencv_focal_length
){
    /*
    This is effectively the opencv fisheye model, which is a polynomial approximation of the fisheye distortion.
    The difference here is that the focal length is not included in the model, so the function is only dependent on the radial distance r.
    */
    double r_scaled = r / opencv_focal_length;
    double theta = atanf(r_scaled);
    double theta2 = theta * theta;
    double theta4 = theta2 * theta2;
    double theta6 = theta4 * theta2;
    double theta8 = theta4 * theta4;
    double theta10 = theta8 * theta2;
    double theta12 = theta6 * theta6;
    double poly = (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12);
    double theta_d = theta*poly;
    if (std::abs(r_scaled) < 1e-10){
        return (1.0 - r_scaled*r_scaled / 3.0)*poly;
    }
    return theta_d/r_scaled;
}

/// \brief This is the two parameter division model, which is a 
///        simple division of the radial distance by a polynomial function.
double division_model_polynomial(double r, double k1, double k2){
    return 1.0 / (1.0 + k1 * r * r + k2 * r * r * r * r);
}


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

    m.def("opencv_fisheye_polynomial", &opencv_fisheye_polynomial, "A function that calculates the opencv fisheye polynomial",
        py::arg("r"),
        py::arg("k1"),
        py::arg("k2"),
        py::arg("k3"),
        py::arg("k4"),
        py::arg("k5") = 0.0,
        py::arg("k6") = 0.0,
        py::arg("opencv_focal_length") = DEFAULT_OPENCV_FOCAL_LENGTH
    );

    m.def("division_model_polynomial", &division_model_polynomial, "A function that calculates the division model polynomial",
        py::arg("r"),
        py::arg("k1"),
        py::arg("k2")
    );

    m.def("undistortDivisionModel", &undistortDivisionModel, "A function that undistorts an image using the division model",
        py::arg("result"),
        py::arg("input_image_bytes"),
        py::arg("d1"),
        py::arg("d2"),
        py::arg("cx"),
        py::arg("cy"),
        py::arg("w"),
        py::arg("h"),
        py::arg("image_amplification_factor") = 3
    );
}
