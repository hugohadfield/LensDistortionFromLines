from typing import Tuple, Dict, Any

import numpy as np


from lens_distortion_pybind import (
    UndistortionResult, 
    processFile, 
    processImageFromBytes,
    opencv_fisheye_polynomial, 
    division_model_polynomial,
    undistortDivisionModel as undistortDivisionModelCpp,
)


def manually_unpack_image_pixel(image_bytes: np.ndarray, x, y, width_, height_, channel):
    """
    Manually unpack the pixel data from the flat image data returned by the C++ code.
    """
    return image_bytes[x+y*width_+channel*width_*height_]


def unpack_image_from_list(image_bytes: np.ndarray, width_: int, height_: int):
    """
    Manually unpack the image data from the flat image data returned by the C++ code.
    """
    output_image = np.zeros((height_, width_, 3), dtype=np.uint8)
    for y in range(height_):
        for x in range(width_):
            for c in range(3):
                output_image[height_ - 1 - y, x, c] = manually_unpack_image_pixel(image_bytes, x, y, width_, height_, c)
    return output_image


def unpack_image_from_list_numpy(image_bytes: np.ndarray, width_: int, height_: int):
    """
    This is the numpy functionality that does the same as `unpack_image_from_list`.
    """
    # Reshape the flat image data to the required shape (height, width, channels)
    reshaped_image = image_bytes.reshape((3, height_, width_))
    # Swap the first and last axes to get the correct shape (height, width, channels)
    reshaped_image = np.swapaxes(reshaped_image, 0, 2)
    reshaped_image = np.swapaxes(reshaped_image, 0, 1)
    # Reverse the rows to match the `height_ - 1 - y` transformation
    output_image = np.ascontiguousarray(reshaped_image[::-1, :, :], dtype=np.uint8)
    return output_image


def process_image_numpy(
    rgb_numpy: np.ndarray,
    output_dir: str = "", 
    write_intermediates: bool = False, 
    write_output: bool = False,
    canny_high_threshold: float = 0.8,
    initial_distortion_parameter: float = 0.0,
    final_distortion_parameter: float = 3.0,
    distance_point_line_max_hough: float = 3.0,
    angle_point_orientation_max_difference: float = 10.0,
    max_lines: int = 100,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    This function processes an image in numpy format.
    """
    if len(output_dir) == 0:
        output_folder = './'
    else:
        output_folder = str(output_dir) + '/'
    tmodel = 'div'
    s_opt_c = 'True'

    height, width, _ = rgb_numpy.shape

    image_flat = pack_rgb_for_cpp(rgb_numpy)
    res = UndistortionResult()
    processImageFromBytes(
        res, 
        image_flat, 
        str(output_folder), 
        width, 
        height, 
        canny_high_threshold,
        initial_distortion_parameter,
        final_distortion_parameter,
        distance_point_line_max_hough,
        angle_point_orientation_max_difference,
        tmodel,
        s_opt_c,
        write_intermediates,
        write_output,
        max_lines
    )
    res_dict = {
        'success': res.success,
        'tmodel': res.tmodel,
        'opt_c': res.opt_c,
        'd1': res.d1,
        'd2': res.d2,
        'cx': res.cx,
        'cy': res.cy,
        'width': res.width,
        'height': res.height
    }
    undistorted_data = np.array(res.undistorted(), dtype=np.uint8)
    undistorted_numpy_array = unpack_image_from_list_numpy(undistorted_data, res.width, res.height)
    return undistorted_numpy_array, res_dict


def process_image_file(
    test_image: str, 
    width: int, 
    height: int, 
    output_dir: str = "",
    write_intermediates: bool = False, 
    write_output: bool = False,
    canny_high_threshold: float = 0.8,
    initial_distortion_parameter: float = 0.0,
    final_distortion_parameter = 3.0,
    distance_point_line_max_hough: float = 3.0,
    angle_point_orientation_max_difference: float = 10.0,
    max_lines: int = 100,
):
    """
    This is the main function that calls the C++ code to undistort an image.
    """
    if len(output_dir) == 0:
        output_folder = './'
    else:
        output_folder = str(output_dir) + '/'
    tmodel = 'div'
    s_opt_c = 'True'

    res = UndistortionResult()
    processFile(
        res, 
        str(test_image), 
        str(output_folder), 
        width, 
        height, 
        canny_high_threshold,
        initial_distortion_parameter,
        final_distortion_parameter,
        distance_point_line_max_hough,
        angle_point_orientation_max_difference,
        tmodel,
        s_opt_c,
        write_intermediates,
        write_output,
        max_lines
    )
    res_dict = {
        'success': res.success,
        'tmodel': res.tmodel,
        'opt_c': res.opt_c,
        'd1': res.d1,
        'd2': res.d2,
        'cx': res.cx,
        'cy': res.cy,
        'width': res.width,
        'height': res.height
    }
    undistorted_data = np.array(res.undistorted(), dtype=np.uint8)
    undistorted_numpy_array = unpack_image_from_list_numpy(undistorted_data, res.width, res.height)
    return undistorted_numpy_array, res_dict


def pack_rgb_for_cpp(image_rgb: np.ndarray):
    """
    This packs an rgb image into a flat array for the C++ code, in the format it expects.
    """
    image_rgb_flip = image_rgb[::-1, :, :]
    reshaped_image = np.swapaxes(image_rgb_flip, 0, 2)
    reshaped_image = np.swapaxes(reshaped_image, 1, 2)
    image_bytes = np.ascontiguousarray(reshaped_image.flatten(), dtype=np.uint8)
    return image_bytes


def undistort_division_model(image_rgb: np.ndarray, d1: float, d2: float, cx: int, cy: int):
    """
    This function calls the C++ code to undistort an image using the division model.
    """
    h, w, _ = image_rgb.shape
    image_flat = pack_rgb_for_cpp(image_rgb)

    undistorted_res = UndistortionResult()
    undistortDivisionModelCpp(
        undistorted_res,
        image_flat, 
        d1,
        d2,
        cx,
        cy,
        w,
        h,
        3
    )
    output = undistorted_res.undistorted()
    undistorted_data = np.array(output, dtype=np.uint8)
    undistorted_numpy_array = unpack_image_from_list_numpy(undistorted_data, undistorted_res.width, undistorted_res.height)
    return undistorted_numpy_array


def scale_distortion_coefs(d1: float, d2: float, new_image_scale: float):
    """
    This function adjusts the distortion coefficients for a new image scale.
    """
    return d1/(new_image_scale**2), d2/(new_image_scale**4)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    test_image = "../example/chicago.png"

    # Load the image 
    image = cv2.imread(test_image)
    height, width, _ = image.shape

    # Process the image to get the distortion coefficients
    image_out, res = process_image_numpy(
        image,
        output_dir="../output/",
        write_intermediates=True,
        write_output=True,
    )
    print(res)
    plt.figure()
    plt.imshow(image_out)
    plt.show()

    # Set up parameters for the division model undistortion
    output_dir = "output"
    d1 = res['d1']
    d2 = res['d2']
    cx = res['cx']
    cy = res['cy']

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    undistorted_image = undistort_division_model(image, d1, d2, cx, cy)
    plt.subplot(2, 2, 2)
    plt.imshow(undistorted_image)

    plt.subplot(2, 2, 3)
    d3, d4 = scale_distortion_coefs(d1, d2, 2.0)
    cx2, cy2 = int(cx*2), int(cy*2)
    # Double image size in opencv
    image2 = cv2.resize(image, (width*2, height*2))
    undistorted_image2 = undistort_division_model(image2, d3, d4, cx2, cy2)
    plt.imshow(undistorted_image2)

    d5, d6 = scale_distortion_coefs(d1, d2, 0.5)
    cx_05, cy_05 = int(cx/2.0), int(cy/2.0)
    # Half image size in opencv
    image05 = cv2.resize(image, (int(width/2.0), int(height/2.0)))
    undistorted_image05 = undistort_division_model(image05, d5, d6, cx_05, cy_05)
    plt.subplot(2, 2, 4)
    plt.imshow(undistorted_image05)

    plt.show()
