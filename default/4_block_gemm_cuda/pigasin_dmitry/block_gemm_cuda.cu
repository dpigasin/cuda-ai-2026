#include "block_gemm_cuda.h"

#include <thread>


constexpr int BLOCK_SIZE = 16;

__global__ void kernel(const float* a, const float* b, float* c, size_t n) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float blockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.f;
    for (int block = 0; block < gridDim.x; ++block) {
        blockA[threadIdx.y * BLOCK_SIZE + threadIdx.x] = a[i * n + (block * BLOCK_SIZE + threadIdx.x)];
        blockB[threadIdx.y * BLOCK_SIZE + threadIdx.x] = b[(block * BLOCK_SIZE + threadIdx.y) * n + j];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[threadIdx.y * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    c[i * n + j] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t numElem = a.size();

    std::vector<float> c;
    std::thread t([&](){c.resize(numElem);});

    float *gpuA, *gpuB, *gpuC;
    const size_t numBytes = numElem * sizeof(float);
    cudaMalloc(&gpuA, numBytes);
    cudaMalloc(&gpuB, numBytes);
    cudaMalloc(&gpuC, numBytes);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(n / BLOCK_SIZE, n / BLOCK_SIZE);

    cudaMemcpy(gpuA, a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b.data(), numBytes, cudaMemcpyHostToDevice);
    kernel<<<numBlocks, blockSize>>>(gpuA, gpuB, gpuC, n);
    t.join();
    cudaMemcpy(c.data(), gpuC, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);

    return c;
}
