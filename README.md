
# Python bindings of automatic lens distortion correction

This repository contains Python bindings of the automatic lens distortion correction algorithm implemented in C++ by Miguel Alemán-Flores, Luis Álvarez, Luis Gómez, and Daniel Santana-Cedrés. The original code is available at [IPOL](http://www.ipol.im/pub/pre/130/).
This repository is a fork of the original code, taken from: [LensDistortionFromLines](https://github.com/alicevision/LensDistortionFromLines).

The following changes have been made to the original code:
- Small modifications to the original code to improve quality and compatibility with Python bindings.
- Added Python bindings using pybind11.
- Added Python utility to match OpenCV distortion coefficients to the distortion coefficients estimated by the C++ code.
- Added scripts to handle undistortion of images, passing data between Python and C++ code.


## How to build the python bindings

Use the following commands to build the python bindings:

```
cd python
cmake ..
make
```

This will create a python module called `lens_distortion_pybind` in the `python` directory. You can import this module in your python code and use the functions provided by the C++ code.

## How to use the python bindings

The best way to interact with the python bindings is to use the `lens_distortion_module.py` library. This library provides a high-level interface to the C++ code and includes unpacking to the correct numpy array size.

## Match OpenCV distortion coefficients

The `match_opencv_distortion.py` script can be used to run the C++ code on an image and then match
the distortion coefficients estimated by the C++ code to the distortion coefficients used by OpenCV. It then also provides a function that will use these distortion coefficients to undistort an image using OpenCV. This function is slightly different to the standard OpenCV undistortion procedure as, due to the difference in the way the fisheye polynomials are defined, we need to apply the distortion polynomial to the image centred on the principal undistortion point. This is all handled by the `def undistort_with_matching_opencv` function so you don't need to worry about it. 

### How the matching works
Matching the division model coefficients to the OpenCV fisheye coefficients is not trivial as the divison model and OpenCV fisheye model are not the same. The division model from the paper is a model for **correcting** distortion and is defined as:

```
r = sqrt((u' - c_x)^2 + (v' - c_x)^2)
d(r) = 1 / (1 + d1 * r^2 + d2 * r^4)

u = d(r) * (u' - c_x) + c_x
v = d(r) * (v' - c_y) + c_y
```
where `r` is the distance from the estimated principal point of distortion. A key thing to note here is that this in its current form is a model for **correcting** distortion and is defined in terms of the **distorted** image coordinates `u'` and `v'` and is not a generative model for **applying** distortion.


The OpenCV fisheye model is defined operating on points projected into normalised image space and is a model for **applying** distortion. It operates on points **before** addition of the image centre. For a given focal length `f` the OpenCV fisheye model is defined as:

```
r_theta = sqrt(x^2 + y^2)
theta = atan(r_theta)
theta_d = theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + k4 * theta^8)
s(r_theta) = theta_d / r_theta

u' = s(r_theta) * f * x + c_x
v' = s(r_theta) * f * y + c_y
```

We need to solve a couple of issues here to allow matching of the two models. Firstly, the OpenCV model requires a focal length `f` to be used, which is not present in the division model. Secondly, the OpenCV model is defined in terms of the pre-distrotion normalised image coordinates `x` and `y` and the division model is defined in terms of the distorted image coordinates `u'` and `v'`.

Our goal here is to take known `d1` and `d2` division model coefficients and find the `k1`, `k2`, `k3`, `k4` OpenCV fisheye coefficients that would produce the same distortion. We can simplify the problem by centring the image on `c_x` and `c_y` in advance. This means we need to match the following:
```
r_theta = sqrt(x^2 + y^2)
theta = atan(r_theta)
theta_d = theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + k4 * theta^8)
s(r_theta) = theta_d / r_theta
u' = s(r_theta) * f * x = s(r_theta) * u
v' = s(r_theta) * f * y = s(r_theta) * v
```

```
r^2 = (u'^2 + v'^2) = (s(r_theta) * f * x)^2 + (s(r_theta) * f * y)^2 = f^2 * s(r_theta)^2 * (x^2 + y^2) = f^2 * s(r_theta)^2 * r_theta^2
r = f * s(r_theta) * r_theta
```
and
```
r = sqrt(u'^2 + v'^2)
d(r) = 1 / (1 + d1 * r^2 + d2 * r^4)
u = d(r) * u'
v = d(r) * v'
u^2 + v^2 = d(r)^2 * (u'^2 + v'^2)

f*x = u
f*y = v
u^2 + v^2 = f^2 * (x^2 + y^2)

f^2 * (x^2 + y^2) = d(r)^2 * (u'^2 + v'^2)
f^2 * r_theta^2 = d(r)^2 * r^2
r_theta^2 = d(r)^2 * r^2 / f^2
r_theta = d(r) * r / f
f * r_theta = d(r) * r
```

This means we can work through the following equations to find the OpenCV coefficients that would produce the same distortion as the division model coefficients:
```
r_theta = d(r) * r / f
r = f * s(r_theta) * r_theta
r = f * s(d(r) * r / f) * d(r) * r / f
r = s(d(r) * r / f) * d(r) * r
r * ( s(d(r) * r / f) * d(r) - 1 ) = 0
s(d(r) * r / f) * d(r) - 1 = 0
s(d(r) * r / f) = 1 / d(r) 
```
So our proceedure can be:
1. Generate a set of `r` values from 1 to near the size of our image.
2. Calculate the `d(r)` values for the division model coefficients.
3. Calculate `d(r) * r / f` for each `r` value with an assumed `f` value.
4. Create a cost function that calculates the difference between `s(d(r) * r / f)` and `1 / d(r)` for each `r` value and for a given `k1, k2, k3, k4` parameterisation set of `s`.
5. Use a minimisation algorithm to find the `k1, k2, k3, k4` parameterisation that minimises the cost function.

This is exactly what is done in the utility functions in `python/match_opencv_fisheye.py`.

## Results
Lets run the algorithm on a few images!



___

___

# ORIGINAL README

% Automatic Lens Distortion Correction Using Two Parameter Polynomial and Division Models with iterative optimization


## ABOUT

* Author    : Miguel Alemán-Flores  <maleman@ctim.es>
              Luis Álvarez  <lalvarez@ctim.es>
              Luis Gómez <lgomez@ctim.es>
              Daniel Santana-Cedrés <dsantana@ctim.es>
* Copyright : (C) 2009-2014 IPOL Image Processing On Line http://www.ipol.im/
* License   : CC Creative Commons "Attribution-NonCommercial-ShareAlike" 
              see http://creativecommons.org/licenses/by-nc-sa/3.0/es/deed.en

## OVERVIEW

This source code provides an implementation of a lens distortion correction algorithm 
algorithm as described in IPOL

http://www.ipol.im/pub/pre/130

This program reads an input image and automatically estimates a 2 parameter 
polynomial or division lens distortion model, that it is used to correct image
distortion, after an iterative optimization process. 

The programs produces 4 outputs: 
   (1) The estimated distortion parameters. 
   (2) An output image with the results of Canny Edge Detector.
   (3) An output image with the estimated distorted lines.
   (4) An output image where the distortion model is applied to correct the 
       image distortion. 

## REQUIREMENTS

The code is written in ANSI C, and should compile on any system with
an ANSI C compiler.

The libpng header and libraries are required on the system for
compilation and execution. See http://www.libpng.org/pub/png/libpng.html

MacOSX already provides libpng in versions before Montain Lion.
However, if you have problems with png library, you can follow the next steps:
	- Install Hombrew (http://brew.hs) and run the commands:
		* brew doctor
		* brew update
		* brew install libpng
		* brew link libpng --force

## COMPILATION

We have checked our code on:
	- Fedora 11 (Leonidas) with GCC version 4.4.1-2
	- Ubuntu 14.04 LTS (trusty) with GCC version 4.8.2
	- Windows 7 with Cygwin 6.1 and GCC version 4.8.2
	- MacOSX 10.6.8 (Snow Leopard) Darwing Kernel version 10.8.0 wiht GCC version 4.2.1

Simply use the provided makefile, with the command `make`.
If you want to use the OpenMP library, please use `make OMP=1`.

Alternatively, you can manually compile
    g++ -Wall -Wextra -O3 lens_distortion_correction_2p_iterative_optimization.cpp 
		ami_primitives/subpixel_image_contours.cpp ami_lens_distortion/lens_distortion_procedures.cpp 
		ami_primitives/line_extraction.cpp ami_lens_distortion/lens_distortion_model.cpp 
		ami_primitives/line_points.cpp ami_image/io_png/io_png.cpp ami_lens_distortion/lens_distortion.cpp 
		ami_utilities/utilities.cpp ami_pol/ami_pol.cpp 
		-lpng -lm -o lens_distortion_correction_2p_iterative_optimization
		
		Or the following alternative compilation line for using the OpenMP library
		
		g++ -Wall -Wextra -O3 -fopenmp lens_distortion_correction_2p_iterative_optimization.cpp 
		ami_primitives/subpixel_image_contours.cpp ami_lens_distortion/lens_distortion_procedures.cpp 
		ami_primitives/line_extraction.cpp ami_lens_distortion/lens_distortion_model.cpp 
		ami_primitives/line_points.cpp ami_image/io_png/io_png.cpp ami_lens_distortion/lens_distortion.cpp 
		ami_utilities/utilities.cpp ami_pol/ami_pol.cpp 
		-lpng -lm -o lens_distortion_correction_2p_iterative_optimization

## USAGE

This program takes 10 parameters:
“exe_file  input_directory output_directory high_threshold_Canny initial_distortion_parameter final_distortion_parameter distance_point_line_max_hough angle_point_orientation_max_difference type_of_lens_distortion_model center_optimization primitives_file” 

* 'exe_file '                             : exe file (called ./lens_distortion_correction_2p_iterative_optimization) 
* 'input_directory'                       : input directory
* 'output_directory'                      : output directory
* 'high_threshold_Canny'                  : float value for the high threshold of the Canny method (between 0.7 and 1.0)
* 'initial_distortion_parameter'          : float value for the initial normalized distortion parameter (greater or equal to -0.5)
* 'final_distortion_parameter'            : float value for the final normalized distortion parameter (greater or equal to the initial value)
* 'distance_point_line_max_hough'         : maximum distance allowed between points and associated lines
* 'angle_point_orientation_max_difference': maximum difference (in degrees) of the point orientation angle and the line angle
* 'type_of_lens_distortion_model'         : type of the lens distortion model for the correction of the distortion (pol or div)
* 'center_optimization'                   : optimization of the center of the lens distortion model (True or False)


## SOURCE CODE ORGANIZATION

The source code is organized in the following folders : 

* 'ami_pol' 		        : polynomial roots computation library.
* 'ami_filters' 		    : basic image filters including Gaussian convolution, gradient, 
					                Canny edge detector, etc.
* 'ami_image'			      : objects and basic methods to manage images.
* 'ami_image_draw'		  : basic procedure to draw primitives in an image.
* 'ami_lens_distortion'	: objects, methods and procedures to manage lens distortion models.
* 'ami_primitives'		  : basic objects and methods to manage primitives (lines and points)
* 'ami_utilities'		    : some auxilary macros and functions. 
* 'documentation'       : doxygen documentation of the source code.
* 'example'             : example input image and results.

## EXAMPLE

You can test the program with the provided test image (building.png) in the 
following way:

./lens_distortion_correction_2p_iterative_optimization /data/input/ /data/output/ 0.8 0.0 3.0 3.0 10.0 div True

Furthermore, you can compare your results with the results present inside the folder 'example'

