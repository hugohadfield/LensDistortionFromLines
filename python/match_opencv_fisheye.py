
from typing import Callable, Dict
import json

import click
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from lens_distortion_module import (
    process_image_file, 
    opencv_fisheye_polynomial as opencv_fisheye_polynomial_pybind,
    division_model_polynomial as division_model_polynomial_pybind
)


DEFAULT_OPENCV_FOCAL_LENGTH = 1000.0


def opencv_fisheye_polynomial_python(
    r: float, 
    k1: float, 
    k2: float, 
    k3: float, 
    k4: float, 
    k5: float = 0.0, 
    k6: float = 0.0,
    opencv_focal_length: float = DEFAULT_OPENCV_FOCAL_LENGTH
) -> float:
    """
    This is effectively the opencv fisheye model, which is a polynomial approximation of the fisheye distortion.
    The difference here is that the focal length is not included in the model, so the function is only dependent on the radial distance r.
    """
    scale: float = opencv_focal_length
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


def opencv_fisheye_polynomial(r: float, k1: float, k2: float, k3: float, k4: float, opencv_focal_length: float = DEFAULT_OPENCV_FOCAL_LENGTH) -> float:
    """
    This is the opencv fisheye polynomial model, which is a polynomial approximation of the fisheye distortion.
    """
    return opencv_fisheye_polynomial_pybind(r, k1, k2, k3, k4, 0.0, 0.0, opencv_focal_length)


def invert_model(model_func: Callable, max_r: float = 600.0) -> Callable:
    """
    Given a model function f(r) that takes a radial distance as input and returns a scaling factor,
    this function will return a function s(r) that also takes a radial distance as input, and returns
    a scaling factor as output such that s(f(r)*r) = r.
    """
    r_array = np.linspace(1.0, max_r, 5000)
    y = np.array([model_func(r) for r in r_array])
    output_rs = y*r_array
    # Now calculate the original r values as a fraction of the output r values
    inverted_scaling = r_array/output_rs
    # Fit PCHIP interpolator from scipy
    inv_func = PchipInterpolator(output_rs, inverted_scaling, extrapolate=True)
    return inv_func


def division_model_polynomial_python(r: float, k1: float, k2: float):
    """
    This is the two parameter division model, which is a simple division of the radial distance by a polynomial function.
    """
    return 1.0 / (1.0 + k1 * r * r + k2 * r * r * r * r)


def match_opencv_to_distortion_model(r_model_func: Callable, max_r: float):
    """
    Match an opencv fisheye radial distortion model to a given radial distortion 
    model implemented as a function of the radial distance in pixels from the distortion centre.
    """
    # Create a range of radial distances to test the model
    r_array = np.linspace(1.0, max_r, 4000)

    # Evaluate the model at the radial distances
    r_model = np.array([r_model_func(r) for r in r_array])

    # Now we need to fit the opencv model to the r_model
    def cost_function(k_array):
        k1, k2, k3, k4 = k_array
        sumout = 0.0
        for i, r in enumerate(r_array):
            sumout += (r_model[i] - opencv_fisheye_polynomial(r, k1, k2, k3, k4))**2
        return sumout
    
    # Initial guess for the parameters
    k0 = [0.0, 0.0, 0.0, 0.0]

    # Run the optimization
    res = minimize(
        cost_function, 
        k0, 
        method='l-bfgs-b',
        options={'ftol': 1e-10, 'disp': False, 'maxiter': 10000, 'pgtol': 1e-8},
    )
    print(res)
    return res.x


def match_opencv_distortion_to_undistortion_model(undistortion_model: Callable, max_r: float):
    """
    s(d(r) * r / f) = 1 / d(r) 
    """
    r_array = np.linspace(1.0, max_r, 4000)
    d = np.array([undistortion_model(r) for r in r_array])
    def cost_function(k_array):
        k1, k2, k3, k4 = k_array
        sumout = 0.0
        for i, r in enumerate(r_array):
            r_theta = d[i] * r / DEFAULT_OPENCV_FOCAL_LENGTH
            sumout += (1.0 / d[i] - opencv_fisheye_polynomial(r_theta, k1, k2, k3, k4, opencv_focal_length=1.0))**2
        return sumout
    k0 = [0.0, 0.0, 0.0, 0.0]
    res = minimize(
        cost_function,
        k0,
        method='l-bfgs-b',
        options={'ftol': 1e-10, 'disp': False, 'maxiter': 10000, 'pgtol': 1e-8},
    )
    print(res)
    return res.x


def generate_opencv_distortion_coefs(
    division_coef_k1: float, 
    division_coef_k2: float, 
    cx: float,
    cy: float,
    max_r: float = 600.0,
    mode = 'direct'
):
    """
    Generate the opencv distortion coefficients from the division model coefficients.
    Also generate a pseudo camera intrinsic matrix for the opencv model.
    """
    if mode == 'direct':
        k_array = match_opencv_distortion_to_undistortion_model(
            lambda r: division_model_polynomial_python(r, division_coef_k1, division_coef_k2),
            max_r
        )
    elif mode == 'inverse':
        ipol_undistortion_model = lambda r: division_model_polynomial_pybind(r, division_coef_k1, division_coef_k2)
        ipol_distortion_model = invert_model(ipol_undistortion_model, max_r=max_r)
        k_array = match_opencv_to_distortion_model(ipol_distortion_model, max_r)
    else:
        raise ValueError(f'Invalid mode for generate_opencv_distortion_coefs: {mode}')
    print(f'k_array: {k_array}')
    K = np.array([
        [DEFAULT_OPENCV_FOCAL_LENGTH, 0, cx],
        [0, DEFAULT_OPENCV_FOCAL_LENGTH, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return k_array, K


def undistort_with_matching_opencv(
    img: np.ndarray,
    K: np.ndarray,
    opencv_k1: float,
    opencv_k2: float,
    opencv_k3: float,
    opencv_k4: float
):
    """
    Undistort an image using the opencv fisheye model with the given distortion coefficients.
    This is slightly different to the standard opencv model as we have to shift the image
    to the principal point before undistorting.
    """
    # Pad the image on all sides by max(cx - w/2, cy - h/2) to avoid black borders
    cx = K[0, 2]
    cy = K[1, 2]
    h, w = img.shape[:2]
    diff_cx = cx - w/2.0
    diff_cy = cy - h/2.0
    pad = int(max(abs(diff_cx), abs(diff_cy)))
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
    Kcopy = K.copy()
    Kcopy[0, 2] = w/2.0
    Kcopy[1, 2] = h/2.0
    k_array = np.array([opencv_k1, opencv_k2, opencv_k3, opencv_k4])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(Kcopy, k_array, np.eye(3), Kcopy, (w, h), cv2.CV_16SC2)
    img_res = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_DEFAULT)
    return img_res, Kcopy
    

def match_and_undistort_with_opencv(
    img: np.ndarray,
    division_coef_k1: float, 
    division_coef_k2: float, 
    cx: float,
    cy: float,
    max_r: float = 600.0,
    mode = 'direct'
):
    """
    Match the opencv fisheye model to a division model and undistort an image.
    """
    k_array, K = generate_opencv_distortion_coefs(
        division_coef_k1,
        division_coef_k2,
        cx,
        cy,
        max_r,
        mode
    )
    img_res, K_new = undistort_with_matching_opencv(
        img,
        K,
        k_array[0],
        k_array[1],
        k_array[2],
        k_array[3]
    )
    return img_res, k_array, K_new


def analyse_division_opencv_fit(res_dict: Dict, max_r: float = 600.0):
    # These are the coefficients and centre of the division model
    division_coefficients = [res_dict['d1'], res_dict['d2']]
    division_centre = [res_dict['cx'], res_dict['cy']]

    # Genenerate the opencv distortion coefficients and camera matrix
    # that might match the division model
    k_array, K = generate_opencv_distortion_coefs(
        division_coefficients[0],
        division_coefficients[1],
        division_centre[0],
        division_centre[1],
        max_r=max_r
    )

    # Plot an graph of how well we match the models
    ipol_undistortion_model = lambda r: division_model_polynomial_pybind(r, division_coefficients[0], division_coefficients[1])
    ipol_distortion_model = invert_model(ipol_undistortion_model, max_r=max_r)
    r_array = np.linspace(1.0, max_r, 1000)
    r_model = np.array([ipol_distortion_model(r) for r in r_array])
    r_res = np.array([opencv_fisheye_polynomial(r, *k_array) for r in r_array])
    
    plt.figure()
    plt.plot(r_array, r_model, label='Division model')
    plt.plot(r_array, r_res, label='OpenCV')
    plt.legend()
    plt.show()


def side_by_side_images(
    img_original: np.ndarray, img_divison: np.ndarray, img_opencv_direct: np.ndarray, img_opencv_inverse: np.ndarray
):
    """
    Display images side by side as sub figures
    """
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_divison, cv2.COLOR_BGR2RGB))
    plt.title('Division model')
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_opencv_direct, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV direct match')
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img_opencv_inverse, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV inverse match')
    plt.tight_layout()
    plt.show()


@click.command()
@click.option('--test_image_name', type=str, default='../example/chicago.png')
@click.option('--output_dir', type=str, default='../output/')
@click.option('--write_intermediates', type=bool, default=False)
@click.option('--write_output', type=bool, default=False)
def cli(test_image_name, output_dir, write_intermediates, write_output):
    # Load a test image
    img = cv2.imread(test_image_name)
    h, w = img.shape[:2]

    # The maximum radial distance to consider for the distortion model, if this is too
    # large there will be a problem with the output. Here we will write the output
    # as well as the intermediate images to allow for debugging.
    max_r = w/2.0 + 10.0
    undistorted_numpy_array, res_dict = process_image_file(
        test_image_name, w, h, 
        str(Path(output_dir).resolve()),
        write_intermediates=write_intermediates, 
        write_output=write_output
    )
    # undistorted_numpy_array from rgb to bgr
    undistorted_numpy_array = cv2.cvtColor(undistorted_numpy_array, cv2.COLOR_RGB2BGR)

    print("The result of undistortion is:")
    print(res_dict)

    # Do some analysis of the results
    analyse_division_opencv_fit(res_dict, max_r=max_r)

    # Undistort the image
    img_opencv_direct, _, _ = match_and_undistort_with_opencv(
        img,
        res_dict['d1'],
        res_dict['d2'],
        res_dict['cx'],
        res_dict['cy'],
        max_r=max_r,
        mode='direct'
    )

    img_opencv_inverse, _, _ = match_and_undistort_with_opencv(
        img,
        res_dict['d1'],
        res_dict['d2'],
        res_dict['cx'],
        res_dict['cy'],
        max_r=max_r,
        mode='inverse'
    )

    # Display the images side by side
    side_by_side_images(img, undistorted_numpy_array, img_opencv_direct, img_opencv_inverse)

    # Save the images
    input_image_name = str(Path(test_image_name).name).split('.')[0]
    output_path = Path('../output/' + input_image_name + '/').resolve()
    output_path.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path) + '/original.png', img)
    cv2.imwrite(str(output_path) + '/division_model.png', undistorted_numpy_array)
    cv2.imwrite(str(output_path) + '/opencv_model.png', img_opencv_direct)

    # Save the results dictionary as json
    with open(str(output_path) + '/results.json', 'w') as f:
        json.dump(res_dict, f)
    

if __name__ == '__main__':
    cli()
    # Example usage: 
    # python match_opencv_fisheye.py --test_image_name ../example/chicago.png --output_dir ../output/ --write_intermediates False --write_output False