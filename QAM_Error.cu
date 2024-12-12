#include <iostream>
using namespace std; 


__global__ void HammingCodes()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    // Define the size of the 2D array
    const int rows = 10;
    const int cols = 10;
    int dataArray[rows][cols];  // Declare a 2D array of size 10x10

    // Seed the random number generator
    srand(time(0));

    // Fill the 2D array with random values (0 or 1)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dataArray[i][j] = rand() % 2;  // Assign 0 or 1 randomly
        }
    }

    // Output the 2D array to check the result
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << dataArray[i][j] << " ";  // Print each element
        }
        cout << endl;  // New line after each row
    }

    return 0;
}
