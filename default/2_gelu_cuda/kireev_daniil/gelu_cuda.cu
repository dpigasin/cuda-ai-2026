#include "gelu_cuda.h"

#include <cuda/std/cmath>

__global__ void GeluCUDAImpl(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        auto v = data[idx];
        data[idx] = v - v / (cuda::std::expf(1.59576912f * v * (1.f + 0.044715f * v * v)) + 1.f);
    }
}

inline constexpr size_t THREADS_NUM = 256; // not the best name

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int isize = input.size();
    const int bytes_size = isize * sizeof(float);
    const float* data_ptr = input.data();
    std::vector<float> result(isize);

    float* res_buff = nullptr;
    cudaMalloc(&res_buff, bytes_size);
    cudaMemcpy(res_buff, data_ptr, bytes_size, cudaMemcpyDefault);

    int blocks = cuda::std::ceil(isize / THREADS_NUM);
    GeluCUDAImpl<<<blocks, THREADS_NUM>>>(res_buff, isize);
    
    float* res_ptr = result.data();

    cudaDeviceSynchronize();
    cudaMemcpy(res_ptr, res_buff, bytes_size, cudaMemcpyDefault);
    cudaFree(res_buff);

    return result;
}