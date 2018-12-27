cmake_minimum_required (VERSION 3.5)

project(Yolo)

list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# environment variable InferenceEngine_DIR can be use instead of relaive path to specify location of configuration file
get_filename_component(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../share" ABSOLUTE)

message(STATUS "Looking for inference engine configuration file at: ${CMAKE_PREFIX_PATH}")
find_package(InferenceEngine 1.2)

if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()


set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)
set(LIB_DL dl)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")

include(feature_defs OPTIONAL)


# Properties->C/C++->General->Additional Include Directories
include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader
    ${CMAKE_CURRENT_SOURCE_DIR}/common/samples
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    )
    

# Find OpenCV libray if exists
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

add_subdirectory(yolo)
add_subdirectory(object_detection_demo_yolov2)
add_subdirectory(common/format_reader)
add_subdirectory(extension)