project(cuda_gemm)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

set(CUDA_NVCC_FLAGS -arch=sm_61;-O3;-G;-g;-std=c++11)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

set_source_files_properties(${CURRENT_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_executable(cuda_gemm ${CURRENT_HEADERS} ${CURRENT_SOURCES} )
