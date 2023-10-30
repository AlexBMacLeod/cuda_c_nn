//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>


#include "layer.h"
#include "activation_functions.cuh"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

void relu(layer* layer)
{
    int dimX = ceil(layer->out / 32.0);
    int dimY = ceil(layer->batch_size / 32.0);
    dim3 dimGrid(ceil( layer->out / 32.0), 1, 1);
    dim3 dimBlock(32.0, 1, 1);

    relu_kernel<<<dimGrid, dimBlock>>>(layer->output->memory_d, layer->out, layer->batch_size);

}


void relu_deriv(layer *layer)
{
    int dimX = ceil(layer->out / 32.0);
    int dimY = ceil(layer->batch_size / 32.0);
    dim3 dimGrid(ceil( layer->out / 32.0), 1, 1);
    dim3 dimBlock(32.0, 1, 1);

    relu_deriv_kernel<<<dimGrid, dimBlock>>>(layer->output->memory_d, layer->input->memory_d, layer->out);
}

void tanh(layer *layer)
{
    int dimX = ceil(layer->out / 32.0);
    int dimY = ceil(layer->batch_size / 32.0);
    dim3 dimGrid(dimX, dimY, 1);
    dim3 dimBlock(32.0, 32.0, 1);

    tanh_kernel<<<dimGrid, dimBlock>>>(layer->output->memory_d, layer->out, layer->batch_size);
}

void tanh_deriv(layer *layer)
{
    int dimX = ceil(layer->out / 32.0);
    int dimY = ceil(layer->batch_size / 32.0);
    dim3 dimGrid(dimX, dimY, 1);
    dim3 dimBlock(32.0, 32.0, 1);

    tanh_deriv_kernel<<<dimGrid, dimBlock>>>(layer->output->memory_d, layer->deriv->memory_d, layer->out, layer->batch_size);
}

void softmax(layer *layer)
{
    int dimX = ceil(layer->out / 32.0);
    int dimY = ceil(layer->batch_size / 32.0);
    dim3 dimGrid(dimX, dimY, 1);
    dim3 dimBlock(32.0, 32.0, 1);

    softmax_kernel<<<dimGrid, dimBlock>>>(layer->output->memory_d, layer->out, layer->batch_size)
}

__global__ 
void relu_kernel(float* __restrict__ d,
                            const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < nCols && Row < nRows)
    {
        d[nRows*Col+Row] = fmaxf(d[nRows*Col+Row], 0);
    }
}


__global__ void 
relu_deriv_kernel(int* __restrict__ d_out, const int* __restrict__ d_in,
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
}
__global__ void 
tanh_kernel(float* __restrict__ d,
                            const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < nCols && Row < nRows)
        d[nRows*Col+Row] = tanhf(d[nRows*Col+Row]);
}

__global__ void 
tanh_deriv_kernel(float* __restrict__ output, float* __restrict__ deriv,
                            const unsigned int nRows, const unsigned int nCols)

    {
        const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;
        if (Col < nCols && Row < nRows)
            deriv[nRows*Col+Row] = 1 - __powf(input[nRows*Col+Row]);
    }


__global__ void 
softmax_kernel(float* __restrict__ d,
                            const unsigned int nRows, const unsigned int nCols,
                            const unsigned int batch_size)
    {
        const unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        __shared__ float sigma[batch_size];

        if (Col < nCols && Row < nRows){
            d[Row*batch_size + Col] = __exp(d[Row*batch_size + Col]);
            sigma[ty] += d[Row*batch_size + Col];
        
            __syncthreads();

            d[Row*batch_size + Col] = d[Row*batch_size + Col]/sigma[ty];
        }
    }

