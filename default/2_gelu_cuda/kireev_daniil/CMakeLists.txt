
project(dk_gelu_cuda LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(
    dk_gelu_cuda main.cpp gelu_cuda.cu gelu_cuda.h
)
set_property(
    TARGET dk_gelu_cuda PROPERTY CXX_STANDARD 17
)
target_include_directories(
    dk_gelu_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)