#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring> // For strcpy

using namespace std;

// BitStream class with a fixed-size char array instead of std::string
class BitStream {
public:
    char bitstring[6]; // Using a fixed-size array to store the bitstring

    BitStream() {
        strcpy(bitstring, "000000"); // Initialize with a default value
    }
};

// Symbol class to store amplitude and phase
class Symbol {
public:
    int amplitude;
    int phase;

    Symbol() : amplitude(0), phase(0) {}
};

// RealModulation function (CUDA kernel)
__global__ void RealModulation(BitStream *TBits, Symbol *Sym, numSymbols) {
    int idx = blockIdx.x * blockDim.x + threadIx.x;
    int batchsize = 10;

    if (TBits->bitstring[0] == '0' && TBits->bitstring[1] == '0' && TBits->bitstring[2] == '0') {
        Sym->amplitude = -4;
    }
    else if (TBits->bitstring[0] == '0' && TBits->bitstring[1] == '0' && TBits->bitstring[2] == '1') {
        Sym->amplitude = -3;
    }
    else if (TBits->bitstring[0] == '0' && TBits->bitstring[1] == '1' && TBits->bitstring[2] == '1') {
        Sym->amplitude = -2;
    }
    else if (TBits->bitstring[0] == '0' && TBits->bitstring[1] == '1' && TBits->bitstring[2] == '0') {
        Sym->amplitude = -1;
    }
    else if (TBits->bitstring[0] == '1' && TBits->bitstring[1] == '1' && TBits->bitstring[2] == '0') {
        Sym->amplitude = 1;
    }
    else if (TBits->bitstring[0] == '1' && TBits->bitstring[1] == '1' && TBits->bitstring[2] == '1') {
        Sym->amplitude = 2;
    }
    else if (TBits->bitstring[0] == '1' && TBits->bitstring[1] == '0' && TBits->bitstring[2] == '1') {
        Sym->amplitude = 3;
    }
    else if (TBits->bitstring[0] == '1' && TBits->bitstring[1] == '0' && TBits->bitstring[2] == '0') {
        Sym->amplitude = 4;
    }
}

__global__ void ImaginaryModulation(BitStream *TBits, Symbol *Sym, numSymbols)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchsize = 10;
    if (idx < numSymbols / batchSize) {

        if (TBits->bitstring[3] == '0' && TBits->bitstring[4] == '0' && (TBits->bitstring[5] == '0')){
        Sym->phase = -4;
            }
        else if (TBits->bitstring[3] == '0' && TBits->bitstring[4] == '0' && TBits->bitstring[5] == '1'){
        Sym->phase = -3;
            } 
        else if (TBits->bitstring[3] == '0' && TBits->bitstring[4] == '1' &&TBits->bitstring[5] == '1'){
        Sym->phase = -2;
            } 
        else if (TBits->bitstring[3] == '0' && TBits->bitstring[4] == '1' && (TBits->bitstring[5] == '0')){
        Sym->phase = -1;
            } 
        else if (TBits->bitstring[3] == '1' && TBits->bitstring[4] == '1' && TBits->bitstring[5] == '0'){
        Sym->phase = 1;
            } 
        else if (TBits->bitstring[3] == '1' && TBits->bitstring[4] == '1' && TBits->bitstring[5] == '1'){
        Sym->phase = 2;
            } 
        else if ((TBits->bitstring[3] == '1') && (TBits->bitstring[4] == '0') && (TBits->bitstring[5] == '1')){
        Sym->phase = 3;
            } 
        else if ((TBits->bitstring[3] == '1') && (TBits->bitstring[4] == '0') && (TBits->bitstring[5] == '0')){
        Sym->phase = 4;
            } 
    }
}

__global__ void RealDeModulation(Symbol *Sym, BitStream *RBits, numSymbols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchsize = 10;
    if (idx < numSymbols / batchSize) {
    // Corrected the comparison with the fixed-size array and proper assignment
    if (Sym->amplitude == -4) {
        RBits->bitstring[0] = '0';
        RBits->bitstring[1] = '0';
        RBits->bitstring[2] = '0';
    }
    else if (Sym->amplitude == -3) {
        RBits->bitstring[0] = '0';
        RBits->bitstring[1] = '0';
        RBits->bitstring[2] = '1';
    } 
    else if (Sym->amplitude == -2) {
        RBits->bitstring[0] = '0';
        RBits->bitstring[1] = '1';
        RBits->bitstring[2] = '1';
    } 
    else if (Sym->amplitude == -1) {
        RBits->bitstring[0] = '0';
        RBits->bitstring[1] = '1';
        RBits->bitstring[2] = '0';
    } 
    else if (Sym->amplitude == 1) {
        RBits->bitstring[0] = '1';
        RBits->bitstring[1] = '1';
        RBits->bitstring[2] = '0';
    } 
    else if (Sym->amplitude == 2) {
        RBits->bitstring[0] = '1';
        RBits->bitstring[1] = '1';
        RBits->bitstring[2] = '1';
    } 
    else if (Sym->amplitude == 3) {
        RBits->bitstring[0] = '1';
        RBits->bitstring[1] = '0';
        RBits->bitstring[2] = '1';
    } 
    else if (Sym->amplitude == 4) {
        RBits->bitstring[0] = '1';
        RBits->bitstring[1] = '0';
        RBits->bitstring[2] = '0';
    }
}
}


__global__ void ImaginaryDeModulation(Symbol *Sym, BitStream *RBits, numSymbols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchsize = 10;
    if (idx < numSymbols / batchSize) {
    // Corrected the comparison with the fixed-size array and proper assignment
    if (Sym->phase == -4) {
        RBits->bitstring[3] = '0';
        RBits->bitstring[4] = '0';
        RBits->bitstring[5] = '0';
    }
    else if (Sym->phase == -3) {
        RBits->bitstring[3] = '0';
        RBits->bitstring[4] = '0';
        RBits->bitstring[5] = '1';
    } 
    else if (Sym->phase == -2) {
        RBits->bitstring[3] = '0';
        RBits->bitstring[4] = '1';
        RBits->bitstring[5] = '1';
    } 
    else if (Sym->phase == -1) {
        RBits->bitstring[3] = '0';
        RBits->bitstring[4] = '1';
        RBits->bitstring[5] = '0';
    } 
    else if (Sym->phase == 1) {
        RBits->bitstring[3] = '1';
        RBits->bitstring[4] = '1';
        RBits->bitstring[5] = '0';
    } 
    else if (Sym->phase == 2) {
        RBits->bitstring[3] = '1';
        RBits->bitstring[4] = '1';
        RBits->bitstring[5] = '1';
    } 
    else if (Sym->phase == 3) {
        RBits->bitstring[3] = '1';
        RBits->bitstring[4] = '0';
        RBits->bitstring[5] = '1';
    } 
    else if (Sym->phase == 4) {
        RBits->bitstring[3] = '1';
        RBits->bitstring[4] = '0';
        RBits->bitstring[5] = '0';
    }
}
}

int main() {
    // Host objects
    BitStream hostStream;
    BitStream outstream;
    Symbol hostSymbol;
    int numSymbols = 10;

     // Initialize random seed
     srand(time(0));  // Seed with the current time to get different results each run

     const int numArrays = 10;  // Number of arrays
     const int arraySize = 3;   // Size of each array
 
     // Declare a 2D array to store the arrays
     int arrays[numArrays][arraySize];
 
     // Fill each array with random 0 or 1
     for (int i = 0; i < numArrays; ++i) {
         for (int j = 0; j < arraySize; ++j) {
             arrays[i][j] = rand() % 2;  // Randomly select 0 or 1
         }
     }
 
     // Output the arrays
     for (int i = 0; i < numArrays; ++i) {
         for (int j = 0; j < arraySize; ++j) {
             std::cout << arrays[i][j] << " ";  // Print each element
         }
         std::cout << std::endl;  // New line after each array
     }
 
    // Device pointers
    BitStream *devStream;
    Symbol *devSymbol;

    // Allocate memory on the device
    cudaMalloc((void**)&devStream, sizeof(BitStream));
    cudaMalloc((void**)&devSymbol, sizeof(Symbol));

    // Copy host data to device
    cudaMemcpy(devStream, &hostStream, sizeof(BitStream), cudaMemcpyHostToDevice);
    cudaMemcpy(devSymbol, &hostSymbol, sizeof(Symbol), cudaMemcpyHostToDevice);

    // Launch the kernel
    RealModulation<<<1, 1>>>(devStream, devSymbol);
    ImaginaryModulation<<<1, 1>>>(devStream, devSymbol);
    

    // Check for errors in kernel launch
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(&hostSymbol, devSymbol, sizeof(Symbol), cudaMemcpyDeviceToHost);

    
    cout << "Amplitude: " << hostSymbol.amplitude << endl;
    cout << "Phase: " << hostSymbol.phase << endl; 




    // Free device memory
    cudaFree(devStream);
    cudaFree(devSymbol);

}




