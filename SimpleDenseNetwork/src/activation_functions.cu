//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>


#include "Linear.h"
#include "activation_functions.cuh"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}


struct activation relu = {
        .func = relu_func,
        .deriv = relu_deriv
};

void relu_func(struct linearLayer* layer)
{
    float *d_Mout;
    float *d_Min;
    int size = layer->out *sizeof(float);


    CHECK_ERROR(cudaMalloc((void**)&d_Mout, size));
    CHECK_ERROR(cudaMalloc((void**)&d_Min, size));


    cudaMemcpy(d_Min, layer->output, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil( layer->out / 32.0), 1, 1);
    dim3 dimBlock(32.0, 1, 1);


    relu_kernel<<<dimGrid, dimBlock>>>(d_Mout, d_Min, layer->out);

    cudaMemcpy(d_Mout, layer->output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_Mout);
    cudaFree(d_Min);
}


void relu_deriv(struct linearLayer*)
{
    float *d_Mout;
    float *d_Min;
    int size = layer->out *sizeof(float);


    CHECK_ERROR(cudaMalloc((void**)&d_Mout, size));
    CHECK_ERROR(cudaMalloc((void**)&d_Min, size));


    cudaMemcpy(d_Min, layer->output, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil( layer->out / 32.0), 1, 1);
    dim3 dimBlock(32.0, 1, 1);


    relu_kernel<<<dimGrid, dimBlock>>>(d_Mout, d_Min, layer->out);

    cudaMemcpy(d_Mout, layer->output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_Mout);
    cudaFree(d_Min);
}

__global__ void relu_kernel(float* __restrict__ d_out, const float* __restrict__ d_in,
                            const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Col < nCols)
    {
        if(d_in[Col]>0){
            d_out[Col] = d_in[Col];
        }
        else d_out[Col] = 0.0;
    }
}


__global__ void relu_deriv_kernel(int* __restrict__ d_out, const int* __restrict__ d_in,
                                  const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < nCols && Row < nRows)
    {
        if(d_in[nRows*Col+Row]>0){
            d_out[nRows*Col+Row] = 1;
        }
        else d_out[nRows*Col+Row] = 0;
    }
