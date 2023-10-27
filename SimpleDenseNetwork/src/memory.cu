#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory.cuh"
#include "matrix.h"

void to_device(Matrix* matrix)
{
    int size = matrix->n * sizeof(float);
    if(matrix->memory_d==NULL){
        CHECK_ERROR(cudaMalloc((void **) matrix->memory_d, size));
    }
    CHECK_ERROR(cudaMemcpy(matrix->memory_h, matrix->memory_d, size, cudaMemcpyDeviceToHost));
}

void to_host(Matrix* matrix)
{
    int size = matrix->n * sizeof(float);
    if(matrix->memory_h == NULL){
        matrix->memory_h = (float *)calloc(matrix->n, sizeof(float));
    }
    CHECK_ERROR(cudaMemcpy(matrix->memory_d, matrix->memory_h, size, cudaMemcpyHostToDevice));
}

void free_device(Matrix* matrix)
{
    if(matrix->memory_d!=NULL){
        cudaFree(matrix->memory_d);
        matrix->memory_d = NULL;
    }
}

void free_host(Matrix* matrix)
{
    if(matrix->memory_h!=NULL){
        free(matrix->memory_h);
        matrix->memory_h = NULL;
    }
}

void input_cpu_data(Matrix* matrix, float* input, int n)
{
    if(matrix->memory_h==NULL){
        matrix->memory_h = (float *)(calloc(n, sizeof(float)));
        matrix->n = n;
    }
    assert(matrix->n==n);
    memccpy(matrix->memory_h, input, n, sizeof(float));
}

void input_gpu_data(Matrix* matrix, float* input, int n)
{

    int size = n * sizeof(float);
    if(matrix->memory_d==NULL){
        CHECK_ERROR(cudaMalloc((void **) matrix->memory_d, size));
        matrix->n = n;
    }
    assert(matrix->n == n);
    CHECK_ERROR(cudaMemcpy(matrix->memory_h, matrix->memory_d, size, cudaMemcpyDeviceToHost));
}

void allocate_cpu(Matrix* matrix)
{
    if(matrix->memory_h==NULL){
        matrix->memory_h = (float *)calloc(matrix->n, sizeof(float));
    }else{
        printf("Trying to allocate memory to non-NULL ptr.");
        exit(EXIT_FAILURE);
    }
}

void allocate_gpu(Matrix* matrix)
{
    int size = matrix->n * sizeof(float);
    if(matrix->memory_d==NULL){
        CHECK_ERROR(cudaMalloc((void **) matrix->memory_d, size));
    }else{
        printf("Trying to allocate memory to non-NULL GPU ptr.");
        exit(EXIT_FAILURE);
    }
}