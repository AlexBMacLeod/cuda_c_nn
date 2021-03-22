#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "Linear.h"
#include "../include/nnKernel.cuh"

#define TILE_SIZE 32
#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
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

void matrixVector(struct linearLayer* layer, float *input, float *output)
{
    float *d_Out;
    float *d_Vec;
    float *d_Matrix;
    int sizeMatrix = layer->in * layer->out * sizeof(float);
    int sizeInVec = layer->in * sizeof(float);
    int sizeOutVec = layer->out * sizeof(float);

    CHECK_ERROR(cudaMalloc((**void)&d_Matrix, sizeMatrix));
    CHECK_ERROR(cudaMalloc((**void)&d_Vec, sizeInVec));
    CHECK_ERROR(cudaMalloc((**void)&d_Out, sizeOutVec));

    dim3 dimGrid(ceil(layer->in/32.0), ceil(layer->out/32.0), 1);
    dim3 dimBlock(32.0, 32.0, 1);

    matvec_kernel<<<dimGrid, dimBlock>>>(d_Matrix, d_Vec, d_Out, layer->in, layer->out);

    cudaMemCpy(d_Out, output, sizeOutVec, cudaMemCpyDeviceToHost);

    cudaFree(d_Out);
    cudaFree(d_Vec);
    cudaFree(d_Matrix);
}

void transpose(struct linearLayer* layer)
{
    (float *odata, float *idata, int width, int height)
}

void forwardPass(float *inputRow, layer hiddenLayers[],
    float *yhat, int numHidden)
{
    float *d_Out;
    float *d_Vec;
    float *d_Matrix;
    int sizeMatrix = rows * cols * sizeof(float);
    int sizeVec = cols * sizeof(float);

    dim3 dimGrid(ceil(cols/32.0), ceil(rows/32.0), 1)
    dim3 dimBlock(32.0, 32.0, 1)

    matvec_kernel<<<dimGrid, dimBlock>>>(d_Matrix, d_x, d_p, hiddenLayers[i].input, hiddenLayers[i].output)

    for(int i = 0; i < numHidden; i++)
    {
        int sizeWeights = hiddenLayers[i].input*hiddenLayers[i].output*sizeof(float);
        int sizeOut = hiddenLayers[i].output*sizeof(float);
        int sizeIn = hiddenLayers[i].input*sizeof(float);

        CHECK_ERROR(cudaMalloc((**void)&d_M, sizeWeights));
        CHECK_ERROR(cudaMalloc((**void)&d_x, sizeIn));
        CHECK_ERROR(cudaMalloc((**void)&d_p, sizeOut));

        cudaMemCpy(hiddenLayers[i].weights, d_M, sizeWeights, cudaMemCpyHostToDevice);

        if(i == 0){
            cudaMemCpy(inputRow, d_x, sizeIn, cudaMemCpyHostToDevice);
        } else cudaMemCpy(hiddenLayers[i-1].neurons, d_x, sizeIn, cudaMemCpyHostToDevice);

        matvec_kernel<<<dimGrid, dimBlock>>>(d_M, d_x, d_p, hiddenLayers[i].input, hiddenLayers[i].output, true)

        if(i+1 == numHidden) {cudaMemCpy(d_p, yhat, sizeOut, cudaMemCpyDeviceToHost);
        } else cudaMemCpy(d_p, hiddenLayers[i].neurons, sizeOut, cudaMemCpyDeviceToHost);

        cudaFree(d_M);
        cudaFree(d_x);
        cudaFree(d_p);

    }
}

void backpass(layer *hiddenLayers[], float *yhat, float alpha)
{
  //code clearly needed here
}