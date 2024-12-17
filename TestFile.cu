#include <iostream>
#include <cuda_runtime.h>

// Kernel function to add two arrays
__global__ void addArrays(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 10;
    int size = N * sizeof(int);

    // Allocate memory for arrays on the host
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    // Initialize arrays on the host
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory for arrays on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the kernel with one block and N threads
    addArrays<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
