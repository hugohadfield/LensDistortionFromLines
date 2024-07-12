from typing import List

from pathlib import Path
import time

import lens_distortion_module 
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define the path to the test image
    test_image = Path('../example/rubiks.png').resolve()
    assert test_image.exists(), f'Test image not found: {test_image}'

    # Get the image dimensions
    img = cv2.imread(str(test_image))
    height, width = img.shape[:2]

    # Define the output directory
    output_dir = Path('../output').resolve()
    # Make the output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    undistorted_numpy_array, res_dict = lens_distortion_module.process_image(test_image, output_dir, width, height)
    print(res_dict)
    plt.figure()
    plt.imshow(undistorted_numpy_array)

    plt.show()
