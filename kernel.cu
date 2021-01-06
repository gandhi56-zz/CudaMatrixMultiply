
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

inline float getRandomFloat() {
    return (float)rand() / (float)RAND_MAX;
}

void initMatrix(float* a, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            a[i * cols + j] = /*getRandomFloat()*/ i;
}

void matrixMultiply(float* c, float* a, float* b, const int aRows, const int aCols, const int bRows, const int bCols) {
    assert(aCols == bRows && "invalid dimensions for matrix multiplication!");
    const int cRows = aRows;
    const int cCols = bCols;
    for (int i = 0; i < cRows; ++i) {
        for (int j = 0; j < cCols; ++j) {
            int cIdx = i * cCols + j;
            c[cIdx] = 0.0;
            for (int k = 0; k < aCols; ++k) {
                int aIdx = i * aCols + k;
                int bIdx = k * bCols + j;
                c[cIdx] += a[aIdx] * b[bIdx];
            }
        }
    }
}

__global__
void matrixMultiplyGPU(float* c, float* a, float* b, const int aRows, const int aCols, const int bRows, const int bCols) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    //printf("compute index: (%d, %d)...\n", i, j);
    const int cRows = aRows;
    const int cCols = bCols;
    int idx = i * cCols + j;
    if (idx < cRows * cCols) {
        c[idx] = 0.0;
        for (int k = 0; k < aCols; ++k) {
            int aIdx = i * aCols + k;
            int bIdx = k * bCols + j;
            c[idx] += a[aIdx] * b[bIdx];
        }
    }
}

void printMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
    cudaSetDevice(0);
    srand((float)time(NULL));

    // set dimensions
    const int aRows = 4;
    const int aCols = 3;
    const int bRows = 3;
    const int bCols = 4;
    const int cRows = 4;
    const int cCols = 4;
    
    // allocate space for matrices
    float* a = (float*)malloc(aRows * aCols * sizeof(float));
    float* b = (float*)malloc(bRows * bCols * sizeof(float));
    float* c = (float*)malloc(cRows * cCols * sizeof(float));

    // initialize matrices
    initMatrix(a, aRows, aCols);
    initMatrix(b, bRows, bCols);

    // allocate global memory
    float* device_a;
    float* device_b;
    float* device_c;
    cudaMalloc((void**)&device_a, aRows * aCols * sizeof(float));
    cudaMalloc((void**)&device_b, bRows * bCols * sizeof(float));
    cudaMalloc((void**)&device_c, cRows * cCols * sizeof(float));

    // copy over data from host to device
    cudaMemcpy(device_a, a, aRows * aCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, bRows * bCols * sizeof(float), cudaMemcpyHostToDevice);

    // execute matrix multiplication on device
    //matrixMultiply(c, a, b, aRows, aCols, bRows, bCols);
    dim3 numThreads(cRows, cCols);
    matrixMultiplyGPU << <1, numThreads>> > (device_c, device_a, device_b, aRows, aCols, bRows, bCols);
    cudaDeviceSynchronize();

    // copy over data from host to device
    cudaMemcpy(c, device_c, cRows * cCols * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(a, aRows, aCols);
    printMatrix(b, bRows, bCols);
    printMatrix(c, cRows, cCols);

    free(a);
    free(b);
    free(c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return 0;
}
