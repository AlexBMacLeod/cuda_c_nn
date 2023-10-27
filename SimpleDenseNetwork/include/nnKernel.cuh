#ifndef _NNKERNEL_CUH
#define _NNKERNEL_CUH

__global__ void matrixMult(const float* __restrict__, const float* __restrict__, float*, int, int, int);

__global__ void vecxvec_kernel(const float* __restrict__ d_m, const float* __restrict__ d_x, float * __restrict__ d_p,
                               const unsigned int nRows, const unsigned int nCols);

__global__ void matvec_kernel(const float* __restrict__ d_M, const float* __restrict__ d_x, float * __restrict__ d_p,
                              const unsigned int nRows, const unsigned int nCols);

__global__ void transpose_kernel(float*, float*, int, int);

void matrixVector(float*, float*, float*, int, int);
#endif