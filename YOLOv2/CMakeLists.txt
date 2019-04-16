cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(InferenceEngine 1.2)

set(TARGET_NAME "yolov2")
set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV libray
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()


find_package(gflags REQUIRED)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/samples
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/helper
    ${CMAKE_CURRENT_SOURCE_DIR}/../extension
    ${InferenceEngine_INCLUDE_DIRS}
)

if( BUILD_SAMPLE_NAME AND NOT ${BUILD_SAMPLE_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "SAMPLE ${TARGET_NAME} SKIPPED")
    return()
endif()

if (NOT DEFINED ENV{CPU_EXTENSION_LIB})
  message(FATAL_ERROR "Please set ENV CPU_EXTENSION_LIB with 'export CPU_EXTENSION_LIB=<path-to-libcpu_extension.so>'")
endif()
set (CpuExtension_lib $ENV{CPU_EXTENSION_LIB})
add_library(cpu_extension SHARED IMPORTED)
set_target_properties(cpu_extension PROPERTIES
  IMPORTED_LOCATION $ENV{CPU_EXTENSION_LIB})

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} 
        cpu_extension 
        pthread 
        gflags
        ${InferenceEngine_LIBRARIES}
        ${LIB_DL}
        ${OpenCV_LIBS}
)
