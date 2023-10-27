#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "linear.h"
#include "nnKernel.cuh"

#define TILE_SIZE 32

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

__global__
void matrixMult(const float * __restrict__ M, const float * __restrict__ N, float *P, int j, int k, int l)
{
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float NdsOne[TILE_SIZE][TILE_SIZE];
    __shared__ float NdsTwo[TILE_SIZE][TILE_SIZE];


    float PvalOne = 0.0;
    float PvalTwo = 0.0;


    int bx = blockIdx.x * 2;    int by = blockIdx.y;
    int tx= threadIdx.x;    int ty = threadIdx.y;

    int Col = bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;


    #pragma unroll
    for(int ph=0; ph<ceil(k/(float)TILE_SIZE); ph++)
    {
        Mds[ty][tx] = 0.0;
        NdsOne[ty][tx] = 0.0;
        NdsTwo[ty][tx] = 0.0;

        __syncthreads();

        if(Row < j && (ph * TILE_SIZE + ty) < k)
            Mds[ty][tx] = M[row*k + TILE_SIZE * ph + tx];
        if(Col < l && (ph * bx + tx) < k)
            NdsOne[ty][tx] = N[(ty + ph * TILE_SIZE) * l + Col];
        if(Col + 1 < l && (ph * TILE_SIZE + tx) < k)
            NdsTwo[ty][tx] = N[(ty + ph * TILE_SIZE) * l + Col + TILE_WIDTH];


        __syncthreads();

        #pragma unroll
        for(int k=0; k<TILE_SIZE; k++)
        {
            PvalOne += Mds[ty][k] * NdsOne[k][tx];
            PvalTwo += Mds[ty][k] * NdsTwo[k][tx];

        }
        __syncthreads();
    }

    if(Row < j && Col < l)
        P[Row * l + Col] = PvalOne;
    if(Row < j && Col + TILE_SIZE < l)
        P[Row * l + Col + TILE_SIZE] = PvalTwo;

}

__global__ void vecxvec_kernel(const float* __restrict__ d_m, const float* __restrict__ d_x, float * __restrict__ d_p,
    const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int tid = blockDimx.x * blockIdx.x + threadIdx.x;
    __shared__ float xds[TILE_SIZE];
    float pval = 0.0;

    #pragma unroll
    for(unsigned int m = 0; m < ((nCols + TILE_SIZE -1)/TILE_SIZE); m++)
    {
        if((m * TILE_SIZE + threadIdx.x) < nCols)
        {
            xds[threadIdx.x] = d_x[threadIdx.x + m * TILE_SIZE];
        } else xds[threadIdx.x] = 0.f;
    }
    __syncthreads();

    if ((Row<Width) && (Col<Width)) p[tid] = Pvalue;

    #pragma unroll
    for(unsigned int e = 0; e < TILE_SIZE; e++)
    {
        pval += d_m[t + (e + TILE_SIZE *m) * nRows] * xds[e];
    }
    
    if ((Row<Width) && (Col<Width)) P[Row*Width+Col] = Pvalue;
}


__global__ void matvec_kernel(const float* __restrict__ d_M, const float* __restrict__ d_x, float * __restrict__ d_p, 
    const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int tid = blockDimx.x * blockIdx.x + threadIdx.x;
    __shared__ float xds[TILE_SIZE];
    float pval = 0.0;

    #pragma unroll
    for(unsigned int m = 0; m < ((nCols + TILE_SIZE -1)/TILE_SIZE); m++)
    {
        if((m * TILE_SIZE + threadIdx.x) < nCols)
        {
            xds[threadIdx.x] = d_x[threadIdx.x + m * TILE_SIZE];
        } else xds[threadIdx.x] = 0.f;
    
        __syncthreads();

        #pragma unroll
        for(unsigned int e = 0; e < TILE_SIZE; e++)
        {
            pval += d_M[tid + (e + TILE_SIZE *m) * nRows] * xds[e];
        }
    }
}

__global__ void transpose_kernel(float *odata, float *idata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];


    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

void matrixVector(float* h_Matrix, float* h_Vec, float* h_Out, int in, int out)
{
    float *d_Out;
    float *d_Vec;
    float *d_Matrix;
    int sizeMatrix = in * out * sizeof(float);
    int sizeInVec = in * sizeof(float);
    int sizeOutVec = out * sizeof(float);

    CHECK_ERROR(cudaMalloc((**void)&d_Matrix, sizeMatrix));
    CHECK_ERROR(cudaMalloc((**void)&d_Vec, sizeInVec));
    CHECK_ERROR(cudaMalloc((**void)&d_Out, sizeOutVec));

    cudaMemCpy(h_Matrix, d_Matrix, sizeMatrix, cudaMemCpyHostToDevice);
    cudaMemCpy(h_Vec, d_Vec, sizeInVec, cudaMemCpyHostToDevice);

    dim3 dimGrid(ceil(in/32.0), ceil(out/32.0), 1);
    dim3 dimBlock(32.0, 32.0, 1);

    matvec_kernel<<<dimGrid, dimBlock>>>(d_Matrix, d_Vec, d_Out, in, out);

    cudaMemCpy(d_Out, output, sizeOutVec, cudaMemCpyDeviceToHost);

    cudaFree(d_Out);
    cudaFree(d_Vec);
    cudaFree(d_Matrix);
}

void transpose( float* h_inMatrix, float* h_outMatrix, int in, int out)
{
    float *d_inMatrix;
    float *d_outMatrix;
    int sizeMatrix = in * out * sizeof(float);

    CHECK_ERROR(cudaMalloc((**void)&d_inMatrix, sizeMatrix));
    CHECK_ERROR(cudaMalloc((**void)&d_outMatrix, sizeMatrix));

    cudaMemCpy(h_inMatrix, d_inMatrix, sizeMatrix, cudaMemCpyHostToDevice);

    dim3 dimGrid(ceil(cols/32.0), ceil(rows/32.0), 1);
    dim3 dimBlock(32.0, 32.0, 1);
    transpose_kernel<<<dimGrid, dimBlock>>>( d_outMatrix, d_inMatrix, in, out);

    cudaMemCpy(d_outMatrix, d_inMatrix, sizeMatrix, cudaMemCpyHostToDevice);

    cudaFree(d_inMatrix);
    cudaFree(d_outMatrix);
}
