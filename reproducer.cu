#include "iostream"
#include <chrono>
#include <cuda_runtime.h>
#define max_points 16000204

using namespace std;
using namespace std::chrono;

// Built and tested using cmake with NVHPC 22.5 (CXX) and CUDA 11.7.64

struct specCord{
    double x[2][4][max_points];
    double y[2][4][max_points];
};

// Initialize the coordinates (x = 0, y = 1)
void initspecCord(specCord &specCord){
    // Initialize x = 0 and y = 1
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<max_points;k++){
                specCord.x[i][j][k]=0.0;
                specCord.y[i][j][k]=1.0;
            }
        }
    }
}

__global__ void cudaTask(specCord &specCord){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if thread is out of bounds
    if(tid >= max_points || tid < 0) return;

    // Perform x = y calculations
    specCord.x[0][0][tid] = specCord.y[0][0][tid];
    specCord.x[0][1][tid] = specCord.y[0][1][tid];
    specCord.x[0][2][tid] = specCord.y[0][2][tid];
    specCord.x[0][3][tid] = specCord.y[0][3][tid];
    specCord.x[1][0][tid] = specCord.y[1][0][tid];
    specCord.x[1][1][tid] = specCord.y[1][1][tid];
    specCord.x[1][2][tid] = specCord.y[1][2][tid];
    specCord.x[1][3][tid] = specCord.y[1][3][tid];
}

// Function for testing the results (Probably should pipe the output to a file)
void printX(specCord &specCord){
    for(int i=0; i<max_points; i++){
        printf("%f ", specCord.x[0][0][i]);
    }
}

specCord coords;

int main(){
    // Initialize coords
    initspecCord(coords);

    // // Define Launch Parameters
    dim3 threadsPerBlock(128, 1, 1);
    dim3 grid((max_points/threadsPerBlock.x)+1, 1, 1);

    // Declare GPU memory pointers
    specCord *coords_d;
    unsigned long long coord_size = sizeof(specCord);

    cudaMalloc(&coords_d, coord_size);
    cudaMemcpy(coords_d, &coords, coord_size, cudaMemcpyHostToDevice);

    // Launch Kernel 1000 times
    auto start = high_resolution_clock::now();
    for(int i=0; i<1000; i++){
        cudaTask<<<grid, threadsPerBlock>>>(*coords_d);
        cudaDeviceSynchronize();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count()/1e6 << " seconds" << endl;

    cudaMemcpy(&coords, coords_d, coord_size, cudaMemcpyDeviceToHost);
    cudaFree(coords_d);

    // printX(coords);
}