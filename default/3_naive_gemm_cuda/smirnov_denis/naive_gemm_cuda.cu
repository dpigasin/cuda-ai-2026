#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "naive_gemm_cuda.h"

__global__ void vecNaiveGemm(float* A, float* B, float* C, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int x = idx % n;
        int y = idx / n;
        float sum = 0.f;
        for (int i = 0; i < n; i++) {
            sum += A[y * n + i] * B[i * n + x];
        }
        C[idx] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);
    
    cudaMemcpy(A, aptr, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, bptr, bytes, cudaMemcpyHostToDevice);
    
    size_t threads = 256;
    size_t blocks = (n * n + threads - 1) / threads;
    vecNaiveGemm<<<blocks, threads>>>(A, B, C, n);

    cudaMemcpy(cptr, C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return c;
}

#if 0
std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.f);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aval = aptr[i * n + k];
            for (int j = 0; j < n; j++) {
                cptr[i * n + j] += aval * bptr[k * n + j];
            }
        }
    }

    return c;
}

int main() {
    size_t n = 4096;
    std::vector<float> a(n*n);
    std::vector<float> b(n*n);
    for (size_t i = 0; i < n*n; i++) {
        a[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
        b[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto c = NaiveGemmCUDA(a, b, n);

    auto cref = NaiveGemmRef(a, b, n);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(c[i] - cref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = NaiveGemmCUDA(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
