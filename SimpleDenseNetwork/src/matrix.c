#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>

#include "matrix.h"
#include "memory.cu"

void flatten(Matrix *matrix)
{
    matrix->shape.y = matrix->shape.x*matrix->shape.y*matrix->shape.z;
    matrix->shape.x = 1;
    matrix->shape.z = 1;
}

void free_matrix(Matrix* matrix)
{
    matrix->free_host(matrix);
    matrix->free_device(matrix);
    free(matrix);
}

Matrix* createMatrix( int n, int x, int y, int z)
{
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->n = 0;
    matrix->memory_d = NULL;
    matrix->memory_h = NULL;
    matrix->shape.n = n;
    matrix->shape.x = x;
    matrix->shape.y = y;
    matrix->shape.z = z;
    matrix->free_host = free_host;
    matrix->free_device = free_device;
    matrix->allocate_h = allocate_h;
    matrix->allocate_d = allocate_d;
    matrix->to_device = to_device;
    matrix->to_host = to_host;
    matrix->flatten = flatten;
    matrix->free_device = free_device;
    matrix->free_host = free_host;
    matrix->free_matrix = free_matrix;
    matrix->input_data_h = input_data_h;
    matrix->input_data_d = input_data_d;
    matrix->zero = zero;
    return matrix;
}

