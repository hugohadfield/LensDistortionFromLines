# Ensure we use modern CMake features
cmake_minimum_required(VERSION 3.12)

PROJECT(cameraLensCalibration C CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
endif ()

FIND_PACKAGE(PNG REQUIRED)

file(
        GLOB_RECURSE
        files_header
        src/*.h
)
file(
        GLOB_RECURSE
        files_cpp
        src/*.cpp
)

add_executable(cameraLensCalibration ${files_header} ${files_cpp})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cameraLensCalibration
    ${PNG_LIBRARIES} OpenMP::OpenMP_CXX
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif()
else()
  target_link_libraries(cameraLensCalibration
    ${PNG_LIBRARIES}
  )
endif()

target_include_directories(cameraLensCalibration PUBLIC
  ${PNG_INCLUDE_DIRS}
)

add_subdirectory(extern/pybind11)

# Create the pybind11 module
pybind11_add_module(lens_distortion_pybind python/lens_distortion_pybind.cpp ${files_header} ${files_cpp})

# Link against libpng
target_link_libraries(lens_distortion_pybind PRIVATE ${PNG_LIBRARIES})

# Include the libpng headers
target_include_directories(lens_distortion_pybind PRIVATE ${PNG_INCLUDE_DIRS})

target_compile_definitions(lens_distortion_pybind
                           PRIVATE VERSION_INFO=0.0.1)
