#include <gelu_cuda.h>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

inline constexpr size_t TEST_SIZE = 134217728; // yes it IS important to make it inline because reasons

namespace {
    std::vector<float> GeluRef(const std::vector<float>& input) {
        std::vector<float> result(input.size());
        for (size_t index = 0; index < input.size(); ++index) {
            float x = input[index];
            result[index] = 0.5 * x * (1 + std::tanh(std::sqrt(2.f/M_PI)*(x+0.044715f*std::pow(x,3))));
        }
        return result;
    }
}

int main() {
    std::vector<float> input(TEST_SIZE);

    // maybe, i should make it into a separate function
    // but not today...
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.f, 100.f);
    
    for (size_t n = 0; n < TEST_SIZE; ++n)
        input[n] = dis(gen);


    std::chrono::steady_clock::time_point beginRef = std::chrono::steady_clock::now();
    auto result_ref = GeluRef(input);
    std::chrono::steady_clock::time_point endRef = std::chrono::steady_clock::now();

    std::cout << "ref " << std::chrono::duration_cast<std::chrono::milliseconds>(endRef - beginRef).count() << " ms" << std::endl;

    GeluCUDA(input);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto result = GeluCUDA(input);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Cuda kernel " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    float error = 0.0f;
    for (size_t n = 0; n < TEST_SIZE; ++n) {
        error = std::max(std::abs(result[n] - result_ref[n]), error);
        if (std::isnan(error)) {
            std::cout << "NAN error - index = " << n << " result = " << result[n] << " ref = " << result_ref[n] << std::endl;
            return 1;
        }
    }

    std::cout << "Max error = " << error << std::endl;

    return 0;
}
