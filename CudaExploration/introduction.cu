/* 
This follows the guide provided on https://developer.nvidia.com/blog/even-easier-introduction-cuda/
Any clarifications needed pls refer to the guide
*/

#include <iostream>
#include <math.h>
#include "cuda_runtime.h"               //headers for the cuda methods
#include "device_launch_parameters.h"

// function to add the elements of two arrays
__global__ void add(int n, float* x, float* y) 
/*"__global__"" to declare this as a method to be executed in cuda
    this is known as a kernal
    GPU code is called device code and CPU code is called host code
*/
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20; // 1M elements, << operation is bitwise shift

    //float* x = new float[N];
    //float* y = new float[N];
    float* x, * y;
    cudaMallocManaged(&x, N * sizeof(float));       //allocation of unified memory, addresses that are accessible from both gpu and cpu
    cudaMallocManaged(&y, N * sizeof(float));       // note that the actual location of the data is managed by nvcc which automatically handles the copying of data to and from the gpu

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add <<<1, 1 >>> (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}