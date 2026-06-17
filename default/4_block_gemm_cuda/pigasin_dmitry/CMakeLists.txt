cmake_minimum_required(VERSION 3.10)

project(4_block_gemm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(4_block_gemm_cuda main.cpp block_gemm_cuda.cu)
