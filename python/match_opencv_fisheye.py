
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize
import numba
from scipy.interpolate import PchipInterpolator


@numba.jit(nopython=True)
def opencv_Fr(
    r: float, k1: float, k2: float, k3: float, k4: float, k5: float = 0.0, k6: float = 0.0) -> float:
    """
    This is effectively the opencv fisheye model, which is a polynomial approximation of the fisheye distortion.
    The difference here is that the focal length is not included in the model, so the function is only dependent on the radial distance r.
    """
    scale: float = 500.0
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
    inverte_scaling = r_array/output_rs
    # Fit PCHIP interpolator from scipy
    inv_func = PchipInterpolator(output_rs, inverte_scaling)
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


if __name__ == '__main__':
    # Match the opencv fisheye model to the division model
    ipol_coefficients = [-1.5252154821541331735e-06, -1.2259105349961298202e-12]
    ipol_centre = [523.31006431267951484, 361.77796045674222114]
    # Calculate the inverse model
    ipol_undistortion_model = lambda r: ipol_division_model(r, ipol_coefficients[0], ipol_coefficients[1])
    ipol_distortion_model = invert_model(ipol_undistortion_model)
    # Now run the model forward, then backwards on the result and check that it is the same
    for r in np.linspace(1.0, 500.0, 500):
        undist_scale = ipol_undistortion_model(r)
        dist_scale = ipol_distortion_model(undist_scale*r)
        np.testing.assert_allclose(1, dist_scale*undist_scale, rtol=1e-3)

    max_r = 600.0
    k_array = match_opencv_to_r_model(ipol_distortion_model, max_r)
    print(k_array)

    import matplotlib.pyplot as plt
    r_array = np.linspace(1.0, max_r, 1000)
    r_model = np.array([ipol_distortion_model(r) for r in r_array])
    r_res = np.array([opencv_Fr(r, *k_array) for r in r_array])
    plt.plot(r_array, r_model, label='Division model')
    plt.plot(r_array, r_res, label='OpenCV')
    plt.legend()
    plt.show()

    import cv2
    # Load a test image
    img = cv2.imread('../example/building.png')
    # Get the image dimensions
    h, w = img.shape[:2]
    # Define the pseudo camera intrinsic matrix for opencv
    K = np.array([
        [500.0, 0, ipol_centre[0]],
        [0, 500.0, ipol_centre[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    
    # Get the size of the image
    h, w = img.shape[:2]

    # Generate undistortion and rectification maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, k_array, np.eye(3), K, (w, h), cv2.CV_16SC2)

    # Apply the undistortion using the maps
    img_res = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)

    # Display the images
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.figure()
    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.title('Undistorted')
    plt.show()