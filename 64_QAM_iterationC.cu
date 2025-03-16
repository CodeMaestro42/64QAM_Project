#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <curand_kernel.h>


using namespace std;

struct Symbol {
    float real;  // real part of the complex number (amplitude)
    float imag;  // imaginary part of the complex number (phase)

    // Default constructor
    Symbol() : real(0), imag(0) {}

    // Parametrized constructor
    Symbol(float r, float i) : real(r), imag(i) {}
};

__device__ float round_to_nearest_odd(float value) {
    // Round to the nearest whole number
    int rounded = roundf(value);
    
    // If the rounded value is even, adjust it to the nearest odd integer
    if (rounded % 2 == 0) {
        if (rounded > value) {
            rounded--;  // Round down to the nearest odd
        } else {
            rounded++;  // Round up to the nearest odd
        }
    }
    
    return (float)rounded;
}


__global__ void HammingCodes(int *dataArray, int *parityArray, int *numOfArrays, int *TBits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *numOfArrays) {  // dereference numOfArrays
        int *currentTbits = &TBits[idx * 6];  // 6 elements each bitstream multiply idx by 6
        
        // Extract the data bits for the current idx
        int d1 = dataArray[idx * 3 + 0];
        int d2 = dataArray[idx * 3 + 1];
        int d3 = dataArray[idx * 3 + 2];

        int p1,p2,p3;
        
        // Assign parity bits based on data bits 
        if ((d1 == 0 && d2 == 0 && d3 == 0) || (d1 == 1 && d2 == 1 && d3 == 1)) {
            // data = 000 or 111, parity = 000
            p1 = 0; p2 = 0; p3 = 0;
        }
        else if ((d1 == 0 && d2 == 0 && d3 == 1) || (d1 == 1 && d2 == 1 && d3 == 0)) {
            // data = 001 or 110, parity = 011
            p1 = 0; p2 = 1; p3 = 1;
        }
        else if ((d1 == 0 && d2 == 1 && d3 == 0) || (d1 == 1 && d2 == 0 && d3 == 1)) {
            // data = 010 or 101, parity = 101
            p1 = 1; p2 = 0; p3 = 1;
        }
        else if ((d1 == 0 && d2 == 1 && d3 == 1) || (d1 == 1 && d2 == 0 && d3 == 0)) {
            // data = 011 or 100, parity = 110
            p1 = 1; p2 = 1; p3 = 0;
        }

        // Assign the parity and data bits to the currentTbits array
        currentTbits[0] = p1;  // p1
        currentTbits[1] = p2;  // p2
        currentTbits[2] = d1;  // d1
        currentTbits[3] = p3;  // p3
        currentTbits[4] = d2;  // d2
        currentTbits[5] = d3;  // d3
    }
}

__global__ void Modulation(Symbol *symbols, int* TBits, int numSymbols)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < numSymbols)
    { 
        // Extract the bits for the real (amplitude) and imaginary (phase) parts
        int bit1 = TBits[idx * 6 + 0];    // bit 1 of the bitstring
        int bit2 = TBits[idx * 6 + 1];  // bit 2 of the bitstring
        int bit3 = TBits[idx * 6 + 2];  // bit 3 of the bitstring
        int bit4 = TBits[idx * 6 + 3];    // bit 4 of the bitstring
        int bit5 = TBits[idx * 6 + 4];  // bit 5 of the bitstring
        int bit6 = TBits[idx * 6 + 5];  // bit 6 of the bitstring

        // Determine the real part (amplitude) based on the first three bits
        float real = 0.0f;
        if (bit1 == 0 && bit2 == 0 && bit3 == 0) real = -7.0f;
        else if (bit1 == 0 && bit2 == 0 && bit3 == 1) real = -5.0f;
        else if (bit1 == 0 && bit2 == 1 && bit3 == 1) real = -3.0f;
        else if (bit1 == 0 && bit2 == 1 && bit3 == 0) real = -1.0f;
        else if (bit1 == 1 && bit2 == 1 && bit3 == 0) real = 1.0f;
        else if (bit1 == 1 && bit2 == 1 && bit3 == 1) real = 3.0f;
        else if (bit1 == 1 && bit2 == 0 && bit3 == 1) real = 5.0f;
        else if (bit1 == 1 && bit2 == 0 && bit3 == 0) real = 7.0f;

        // Determine the imaginary part (phase) based on the last three bits
        float imag = 0.0f;
        if (bit4 == 0 && bit5 == 0 && bit6 == 0) imag = -7.0f;
        else if (bit4 == 0 && bit5 == 0 && bit6 == 1) imag = -5.0f;
        else if (bit4 == 0 && bit5 == 1 && bit6 == 1) imag = -3.0f;
        else if (bit4 == 0 && bit5 == 1 && bit6 == 0) imag = -1.0f;
        else if (bit4 == 1 && bit5 == 1 && bit6 == 0) imag = 1.0f;
        else if (bit4 == 1 && bit5 == 1 && bit6 == 1) imag = 3.0f;
        else if (bit4 == 1 && bit5 == 0 && bit6 == 1) imag = 5.0f;
        else if (bit4 == 1 && bit5 == 0 && bit6 == 0) imag = 7.0f;

        // Assign the real and imaginary parts to the symbol struct
        symbols[idx].real = real;
        symbols[idx].imag = imag;
    }
}


__global__ void Demodulation(Symbol* symbols, int* RBits, int numSymbols)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSymbols) {
        // Extract the real and imaginary parts (complex values)
        float real = symbols[idx].real;
        float imag = symbols[idx].imag;

        // Round the real and imaginary values to the nearest whole number
        real = round_to_nearest_odd(real);
        imag = round_to_nearest_odd(imag);

        // Initialize bit values
        int bit1 = 0, bit2 = 0, bit3 = 0;
        int bit4 = 0, bit5 = 0, bit6 = 0;

        // Demodulate the real part (amplitude)
        if (real == -7.0f) {
            bit1 = 0; bit2 = 0; bit3 = 0;
        } 
        else if (real == -5.0f) {
            bit1 = 0; bit2 = 0; bit3 = 1;
        } 
        else if (real == -3.0f) {
            bit1 = 0; bit2 = 1; bit3 = 1;
        } 
        else if (real == -1.0f) {
            bit1 = 0; bit2 = 1; bit3 = 0;
        } 
        else if (real == 1.0f) {
            bit1 = 1; bit2 = 1; bit3 = 0;
        } 
        else if (real == 3.0f) {
            bit1 = 1; bit2 = 1; bit3 = 1;
        } 
        else if (real == 5.0f) {
            bit1 = 1; bit2 = 0; bit3 = 1;
        } 
        else if (real == 7.0f) {
            bit1 = 1; bit2 = 0; bit3 = 0;
        }

        // Demodulate the imaginary part (phase)
        if (imag == -7.0f) {
            bit4 = 0; bit5 = 0; bit6 = 0;
        } 
        else if (imag == -5.0f) {
            bit4 = 0; bit5 = 0; bit6 = 1;
        } 
        else if (imag == -3.0f) {
            bit4 = 0; bit5 = 1; bit6 = 1;
        } 
        else if (imag == -1.0f) {
            bit4 = 0; bit5 = 1; bit6 = 0;
        } 
        else if (imag == 1.0f) {
            bit4 = 1; bit5 = 1; bit6 = 0;
        } 
        else if (imag == 3.0f) {
            bit4 = 1; bit5 = 1; bit6 = 1;
        } 
        else if (imag == 5.0f) {
            bit4 = 1; bit5 = 0; bit6 = 1;
        } 
        else if (imag == 7.0f) {
            bit4 = 1; bit5 = 0; bit6 = 0;
        }

        // Assign the demodulated bits to the RBits array
        RBits[idx * 6 + 0] = bit1;
        RBits[idx * 6 + 1] = bit2;
        RBits[idx * 6 + 2] = bit3;
        RBits[idx * 6 + 3] = bit4;
        RBits[idx * 6 + 4] = bit5;
        RBits[idx * 6 + 5] = bit6;
    }
}

// Device function to generate Gaussian noise
__device__ float gaussian_noise(float mu, float sigma, curandState *state) {
    // Generate two uniform random numbers
    float u0 = curand_uniform(state);  // uniform random between [0, 1)
    float u1 = curand_uniform(state);  // uniform random between [0, 1)

    // Box-Muller transform to generate a standard normal random variable
    float z0 = sqrtf(-2.0f * logf(u0)) * cosf(2.0f * 3.14f * u1);

    // Scale to the desired mean and standard deviation
    return mu + sigma * z0;
}

// Kernel to add Gaussian noise to symbols
__global__ void add_gaussian_noise_to_symbols(Symbol* symbols, int numSymbols, float mu, float sigma, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSymbols) {
        // Add Gaussian noise to the real part of the symbol
        symbols[idx].real += gaussian_noise(mu, sigma, &state[idx]);

        // Add Gaussian noise to the imaginary part of the symbol
        symbols[idx].imag += gaussian_noise(mu, sigma, &state[idx]);
    }
}

// Kernel to initialize the random state
__global__ void init_random_state(curandState* state, unsigned long seed, int numSymbols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSymbols) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

//correction kernel 
/* 
__global__ void ErrorDetect(int numErrors, int RBits)
{ 

}

*/

int main() {
    const int numOfArrays = 4;
    const int rows = numOfArrays;
    const int cols = 3;
    const int numSymbols = numOfArrays;

    int dataArray[rows][cols];  // Declare a 2D array for the data (10x3)
    int parityArray[rows][cols];  // Declare a 2D array for the parity (10x3)
    
    srand(time(0));  // Seed the random number generator

    // Fill the data array with random values (0 or 1)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dataArray[i][j] = rand() % 2;  // Random 0 or 1
        }
    }

    // Fill the parity array with initial values (0)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            parityArray[i][j] = 0;  // Initially zero, will be modified by kernel
        }
    }

    // Output the data array
    cout << "Data Array:" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << dataArray[i][j] << " ";
        }
        cout << endl;
    }

    // CUDA memory allocation
    int *d_DataArray, *d_ParityArray, *d_NumArrays, *d_TBits;

    cudaMalloc((void**)&d_DataArray, sizeof(int) * numOfArrays * 3);  // 3 ints for each data array
    cudaMalloc((void**)&d_ParityArray, sizeof(int) * numOfArrays * 3);  // Parity array (3 per data array)
    cudaMalloc((void**)&d_NumArrays, sizeof(int));  // Only one integer for numOfArrays
    cudaMalloc((void**)&d_TBits, sizeof(int) * numOfArrays * 6);  // 6 ints for each BitStream (Hamming code)


    // Copy data from host to device
    cudaMemcpy(d_DataArray, dataArray, sizeof(int) * numOfArrays * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ParityArray, parityArray, sizeof(int) * numOfArrays * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NumArrays, &numOfArrays, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (numSymbols + threadsPerBlock - 1) / threadsPerBlock;

    HammingCodes<<<blocksPerGrid, threadsPerBlock>>>(d_DataArray, d_ParityArray, d_NumArrays, d_TBits);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    int *TBits_host = new int[numOfArrays * 6];  // Allocate memory for Hamming code (6 ints per array)
    cudaMemcpy(TBits_host, d_TBits, sizeof(int) * numOfArrays * 6, cudaMemcpyDeviceToHost);

     // Output the Hamming code (TBits)
     cout << "Hamming Code (TBits):" << endl;
     for (int i = 0; i < numOfArrays; i++) {
         cout << "BitStream " << i << ": ";
         for (int j = 0; j < 6; j++) {
             cout << TBits_host[i * 6 + j] << " ";
         }
         cout << endl;
     }

    Symbol *d_symbol;

    int numOfSymbols = numOfArrays;

    // Allocate memory on the device for d_symbol
    cudaMalloc((void**)&d_symbol, sizeof(int) * numOfSymbols * 2);
    curandState* d_state;
    cudaMalloc(&d_state, numSymbols * sizeof(curandState));
    //MODULATION   
    Modulation<<<blocksPerGrid,threadsPerBlock>>>(d_symbol,d_TBits,numOfSymbols);
    cudaDeviceSynchronize();

    // Allocate memory on the host for the results
    Symbol* h_symbol = new Symbol[numOfSymbols];

    // Copy the results from device to host
    cudaMemcpy(h_symbol, d_symbol, sizeof(Symbol) * numOfSymbols, cudaMemcpyDeviceToHost);

    // Print the results from host
    for (int i = 0; i < numOfSymbols; i++) {
        std::cout << "Symbol " << i << " Amplitude: " << h_symbol[i].real << std::endl;
    }

    for (int i = 0; i < numOfSymbols; i++) {
        std::cout << "Symbol " << i << " Phase : " << h_symbol[i].imag << std::endl;
    }

    std::cout << "........................." << std::endl;
    std::cout << "Transmission " << std::endl;

    int *d_RBits;
    int *RBits_host = new int[numOfArrays * 6];

    cudaMalloc((void**)&d_RBits,sizeof(int)*numOfArrays*6);
    //channel noise addition 
    // Initialize random states for each thread
    init_random_state<<<(numSymbols + 255) / 256, 256>>>(d_state, 1234, numSymbols);

    // Add Gaussian noise to the modulated symbols
    add_gaussian_noise_to_symbols<<<(numSymbols + 255) / 256, 256>>>(d_symbol, numSymbols, 0.0f, 1.0f, d_state);

    // performed after allocating received bits memory, so noise added memory is same as demodulated memory
    //GaussNoise<<<blocksPerGrid,threadsPerBlock>>>(d_symbol,d_RBits,numOfSymbols);
    cudaDeviceSynchronize();

     //allocate memory for received bits
    //demodulation 
    Demodulation<<<blocksPerGrid,threadsPerBlock>>>(d_symbol,d_RBits,numOfSymbols); 
    //ErrorDC<<<blocksPerGrid,threadsPerBlock>>>();
    cudaMemcpy(RBits_host, d_RBits, sizeof(int) * numOfArrays *6, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numOfArrays; i++) {
        cout << "Bitsream " << i << ": ";
        for (int j = 0; j < 6; j++) {
            cout << RBits_host[i * 6 + j] << " ";
        }
        cout << endl;
    }

    // Copy results back to host
    cudaDeviceSynchronize();
    // Copy modified parityArray back to host
    cudaMemcpy(parityArray, d_ParityArray, sizeof(int) * numOfArrays * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(dataArray, d_DataArray, sizeof(int) * numOfArrays * 3, cudaMemcpyDeviceToHost);
     
    // Free device memory
    cudaFree(d_DataArray);
    cudaFree(d_ParityArray);
    cudaFree(d_NumArrays);
    cudaFree(d_TBits);
    cudaFree(d_symbol);
    cudaFree(d_RBits);
    // Free allocated memory
    delete[] h_symbol;  // Free host memory
    cudaFree(d_symbol);  // Free device memory
    delete[] TBits_host;
    delete[] RBits_host;
    return 0;
}
