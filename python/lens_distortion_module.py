import numpy as np


from lens_distortion_pybind import UndistortionResult, processFile


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


def process_image(
        test_image: str, width: int, height: int, output_dir: str = "",
        write_intermediates: bool = False, write_output: bool = False
    ):
    """
    This is the main function that calls the C++ code to undistort an image.
    """
    if len(output_dir) == 0:
        output_folder = './'
    else:
        output_folder = str(output_dir) + '/'
    canny_high_threshold = 0.8
    initial_distortion_parameter = 0.0
    final_distortion_parameter = 8.0
    distance_point_line_max_hough = 10.0
    angle_point_orientation_max_difference = 10.0
    max_lines = 200
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
        'k1': res.k1,
        'k2': res.k2,
        'cx': res.cx,
        'cy': res.cy,
        'width': res.width,
        'height': res.height
    }
    undistorted_data = np.array(res.undistorted(), dtype=np.uint8)
    undistorted_numpy_array = unpack_image_from_list_numpy(undistorted_data, res.width, res.height)
    return undistorted_numpy_array, res_dict

