/* 
This follows the guide provided on https://developer.nvidia.com/blog/even-easier-introduction-cuda/
Any clarifications needed pls refer to the guide
*/

#include <iostream>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"               //headers for the cuda methods
#include "device_launch_parameters.h"

// function to add the elements of two arrays
__global__ void add(int n, float* x, float* y) 
/*"__global__"" to declare this as a method to be executed in cuda
    this is known as a kernal
    GPU code is called device code and CPU code is called host code
*/
{
    //int index = threadIdx.x; //threadIdx returns the index of the current thread
    //int stride = blockDim.x; //blockDim returns the number of threads in the current block

    int index = blockIdx.x * blockDim.x + threadIdx.x;  //block index * threads per block * thread index
    int increment = blockDim.x * gridDim.x;             //threads per block * total threads active
    for (int i = index; i < n; i += increment) {
		y[i] = x[i] + y[i];
		//printf("%d\n", i);
    }
}

int main(void)
{
    printf("Process started");
    
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

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize; // calculation of the amount of blocks of 256 needed to complete the task
    clock_t tstart = clock();
    // Run kernel on 1M elements on the GPU
    add <<<numBlocks, 256 >>> (N, x, y);
    /* The key code here is the <<< blocks , threads >>> that tells the compiler this code is meant to run on the GPU
    Threads have to be in a multiple of 32, maximum of 1024*/

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("Time taken: %.9fs\n", (double)(clock() - tstart) / CLOCKS_PER_SEC);

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