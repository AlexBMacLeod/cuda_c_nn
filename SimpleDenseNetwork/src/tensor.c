#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>

#include "../include/tensor.h"

static void input(Matrix*, float*);
static void reallocateMem( Matrix*, struct Shape);
static void freeMatrix( Matrix*);

static void reallocateMem(Matrix *matrix, struct Shape shape)
{
    //matrix->shape.x = x;
    //matrix->shape.y = y;
    matrix->cpuData = realloc(matrix->cpuData, sizeof(float)*matrix->shape.x*matrix->shape.y);
}

static void freeCpu(struct Matrix* matrix)
{
    if(matrix->cpuAllocation)
    {
        free(matrix->cpuData);
        matrix->cpuData = NULL;
        matrix->cpuAllocation = 0;
    }
}

static void freeGpu(struct Matrix* matrix)
{
    if(matrix->gpuAllocation)
    {
        cudaFree(matrix->gpuData);
        matrix->gpuData = NULL;
        matrix->gpuAllocation = 0;
    }
}

static void freeMatrix(Matrix *matrix)
{
    matrix->freeCpu(matrix);
    matrix->freeGpu(matrix);
    free(matrix);
}

static void copyMemToCpu(struct Matrix *matrix)
{
    if(matrix->gpuAllocation)
    {
        if(matrix->cpuAllocation)
        {
            int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);
            cudaMemcpy(matrix->cpuData, matrix->gpuData, size, cudamemcpyDeviceToHost);
        }else{
            matrix->allocateCpu(matrix);
            int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);
            cudaMemcpy(matrix->cpuData, matrix->gpuData, size, cudamemcpyDeviceToHost);
        }
    }
}

static void allocateGpu(struct Matrix *matrix)
{
    if(!matrix->gpuAllocation)
    {
        int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);
        CHECK_ERROR(cudaMalloc((void **) matrix->gpuData, size));
        matrix->gpuAllocation = 1;
    }
}

static void allocateCpu(struct Matrix *matrix)
{
    if(!matrix->cpuAllocation)
    {
        int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);

        matrix->cpuData = calloc(matrix->shape.n*matrix->shape.x*matrix->shape.y*matrix->shape.z, sizeof(float));

        matrix->cpuAllocation = 1;
    }
}

static void flatten(Matrix *matrix)
{
    matrix->shape.y = matrix->shape.x*matrix->shape.y*matrix->shape.z;
    matrix->shape.x = 1;
    matrix->shape.z = 1;
}

static void inputCpuData(Matrix* matrix, float* inMatrix)
{
    int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);
    if(gpuAllocation)
        cudaMemcpy(matrix->gpuData, inMatrix, size, cudaMemcpyHostToDevice);
    else{
        matrix->allocateGpu(matrix);
        cudaMemcpy(matrix->gpuData, inMatrix, size, cudaMemcpyHostToDevice);
    }
}

static void inputGpuData(Matrix* matrix, float* inMatrix)
{
    int size = matrix->shape.x*matrix->shape.y*matrix->shape.z*sizeof(float);
    if(gpuAllocation)
        cudaMemcpy(matrix->gpuData, inMatrix, size, cudaMemcpyDeviceToDevice);
    else{
        matrix->allocateGpu(matrix);
        cudaMemcpy(matrix->gpuData, inMatrix, size, cudaMemcpyDeviceToDevice);
    }
}

static void zero(Matrix* matrix)
{
    long data_size = matrix->shape.n*matrix->shape.x*matrix->shape.y*matrix->shape.z * sizeof(float);
    cudaMemset(matrix->gpuData, 0, data_size);
}

Matrix* createMatrix( int n, int x, int y, int z)
{
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->cpuAllocation = 0;
    matrix->gpuAllocation = 0;
    matrix->cpuData = NULL;
    matrix->gpuData = NULL;
    matrix->shape.n = n;
    matrix->shape.x = x;
    matrix->shape.y = y;
    matrix->shape.z = z;
    matrix->freeCpu = freeCpu;
    matrix->freeGpu = freeGpu;
    matrix->allocateCpu = allocateCpu;
    matrix->allocateGpu = allocateGpu;
    matrix->copyMemToCpu = copyMemToCpu;
    matrix->flatten = flatten;
    matrix->freeMem = freeMatrix;
    matrix->inputData = input;
    matrix->zero = zero;
    matrix->allocateGpu(matrix);
    return matrix;
}

