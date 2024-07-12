
from typing import Callable

import numpy as np
from scipy.optimize import minimize
import numba
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import cv2

from lens_distortion_module import process_image


DEFAULT_OPENCV_FOCAL_LENGTH = 1000.0


@numba.jit(nopython=True)
def opencv_Fr(
    r: float, k1: float, k2: float, k3: float, k4: float, k5: float = 0.0, k6: float = 0.0) -> float:
    """
    This is effectively the opencv fisheye model, which is a polynomial approximation of the fisheye distortion.
    The difference here is that the focal length is not included in the model, so the function is only dependent on the radial distance r.
    """
    scale: float = DEFAULT_OPENCV_FOCAL_LENGTH
    r_scaled = r / scale
    theta = np.arctan(r_scaled)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta10 = theta8 * theta2
    theta12 = theta6 * theta6
    poly = (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12)
    theta_d = theta*poly
    if np.abs(r_scaled) < 1e-10:
        return (1.0 - r_scaled*r_scaled / 3.0)*poly
    return theta_d/r_scaled


def invert_model(model_func: Callable, max_r: float = 600.0) -> Callable:
    r_array = np.linspace(1.0, max_r, 5000)
    y = np.array([model_func(r) for r in r_array])
    output_rs = y*r_array
    # Now calculate the original r values as a fraction of the output r values
    inverted_scaling = r_array/output_rs
    # Fit PCHIP interpolator from scipy
    inv_func = PchipInterpolator(output_rs, inverted_scaling, extrapolate=True)
    return inv_func


@numba.jit(nopython=True)
def ipol_division_model(r: float, k1: float, k2: float):
    """
    This is the two parameter division model, which is a simple division of the radial distance by a polynomial function.
    """
    return 1.0 / (1.0 + k1 * r * r + k2 * r * r * r * r)


def match_opencv_to_r_model(r_model_func: Callable, max_r: float):
    """
    Match an opencv fisheye radial distortion model to a given radial distortion 
    model implemented as a function of the radial distance in pixels from the distortion centre.
    """
    # Create a range of radial distances to test the model
    r_array = np.linspace(1.0, max_r, 4000)

    # Evaluate the model at the radial distances
    r_model = np.array([r_model_func(r) for r in r_array])

    # Now we need to fit the opencv model to the r_model
    @numba.jit(nopython=True)
    def cost_function(k_array):
        k1, k2, k3, k4 = k_array
        sumout = 0.0
        for i, r in enumerate(r_array):
            sumout += (r_model[i] - opencv_Fr(r, k1, k2, k3, k4))**2
        return sumout
    
    # Initial guess for the parameters
    k0 = [0.0, 0.0, 0.0, 0.0]

    # Run the optimization
    res = minimize(
        cost_function, 
        k0, 
        method='l-bfgs-b',
        options={'atol': 1e-10, 'disp': False, 'maxiter': 10000, 'rtol': 1e-10},
    )
    print(res)
    return res.x


def generate_opencv_distortion_coefs(
        division_coef_k1: float, 
        division_coef_k2: float, 
        cx: float,
        cy: float,
        max_r: float = 600.0):
    """
    Generate the opencv distortion coefficients from the division model coefficients.
    Also generate a pseudo camera intrinsic matrix for the opencv model.
    """
    ipol_undistortion_model = lambda r: ipol_division_model(r, division_coef_k1, division_coef_k2)
    ipol_distortion_model = invert_model(ipol_undistortion_model, max_r=max_r)
    k_array = match_opencv_to_r_model(ipol_distortion_model, max_r)
    print(f'k_array: {k_array}')
    K = np.array([
        [DEFAULT_OPENCV_FOCAL_LENGTH, 0, cx],
        [0, DEFAULT_OPENCV_FOCAL_LENGTH, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return k_array, K


def match_and_undistort_with_opencv(
        img: np.ndarray,
        division_coef_k1: float, 
        division_coef_k2: float, 
        cx: float,
        cy: float,
        max_r: float = 600.0):
    """
    Match the opencv fisheye model to a division model and undistort an image.
    """
    k_array, K = generate_opencv_distortion_coefs(
        division_coef_k1,
        division_coef_k2,
        cx,
        cy,
        max_r,
    )
    # Pad the image on all sides by max(cx - w/2, cy - h/2) to avoid black borders
    h, w = img.shape[:2]
    diff_cx = cx - w/2.0
    diff_cy = cy - h/2.0
    pad = int(max(diff_cx, diff_cy))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Now shift the image so that the principal point is at the centre, we have to do this
    # because the opencv model is a model with distances relative to the centre of the image
    # and then shifted to the principal point, while the division model is a model with distances
    # defined in image space relative to the principal point.
    img = np.roll(img, -int(diff_cx), axis=1)
    img = np.roll(img, int(diff_cy), axis=0)
    # Now trim the image to the original size
    img = img[pad:pad+h, pad:pad+w]
    # Set the principal point to the centre
    K[0, 2] = w/2.0
    K[1, 2] = h/2.0
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, k_array, np.eye(3), K, (w, h), cv2.CV_16SC2)
    img_res = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_DEFAULT)
    return img_res, k_array, K


if __name__ == '__main__':
    # Load a test image
    test_image_name = '../example/rubiks.png'
    img = cv2.imread(test_image_name)
    h, w = img.shape[:2]
    max_r = w/2.0 + 10.0
    undistorted_numpy_array, res_dict = process_image(test_image_name, w, h)
    print(res_dict)

    # These are the coefficients and centre of the division model
    # ipol_coefficients = [-1.5252154821541331735e-06, -1.2259105349961298202e-12]
    # ipol_centre = [523.31006431267951484, 361.77796045674222114]
    ipol_coefficients = [res_dict['k1'], res_dict['k2']]
    ipol_centre = [res_dict['cx'], res_dict['cy']]

    # Genenerate the opencv distortion coefficients and camera matrix
    # that might match the division model
    k_array, K = generate_opencv_distortion_coefs(
        ipol_coefficients[0],
        ipol_coefficients[1],
        ipol_centre[0],
        ipol_centre[1],
        max_r=max_r
    )

    # Plot an image of how well we match the models
    ipol_undistortion_model = lambda r: ipol_division_model(r, ipol_coefficients[0], ipol_coefficients[1])
    ipol_distortion_model = invert_model(ipol_undistortion_model, max_r=max_r)
    r_array = np.linspace(1.0, max_r, 1000)
    r_model = np.array([ipol_distortion_model(r) for r in r_array])
    r_res = np.array([opencv_Fr(r, *k_array) for r in r_array])
    plt.plot(r_array, r_model, label='Division model')
    plt.plot(r_array, r_res, label='OpenCV')
    plt.legend()

    # Undistort the image
    img_res, k_array, K = match_and_undistort_with_opencv(
        img,
        ipol_coefficients[0],
        ipol_coefficients[1],
        ipol_centre[0],
        ipol_centre[1],
        max_r=max_r
    )

    # Display the images
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.figure()
    plt.imshow(undistorted_numpy_array)
    plt.title('Undistorted division model')
    plt.figure()
    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.title('Undistorted opencv')
    plt.show()