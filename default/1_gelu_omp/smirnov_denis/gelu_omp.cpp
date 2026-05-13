#include <cmath>
#include "gelu_omp.h"

inline float fast_tanh(float x) {
    float e2x = std::exp(2.f * x);
    return (e2x - 1.f) / (e2x + 1.f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float x = inptr[i];
        float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        outptr[i] = 0.5f * x * (1.f + fast_tanh(inner));
    }

    return output;
}
