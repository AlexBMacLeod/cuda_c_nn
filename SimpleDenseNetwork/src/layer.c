#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/layer.h"

static void reallocateMem(Matrix *matrix, int x, int y)
{
    matrix->x = x;
    matrix->y = y;
    matrix->data = realloc(matrix->data, sizeof(float)*matrix->x*matrix->y);
}

static void freeMatrix(Matrix *matrix)
{
    free(matrix->data);
    free(matrix);
}

static void input(Matrix* matrix, float* inMatrix)
{
    memcpy( matrix->data, inMatrix, sizeof(float)*matrix->x*matrix->y);
}

Matrix* createMatrix( int x, int y)
{
    Matrix* matrix = malloc(sizeof(Matrix));
    matrix->x = x;
    matrix->y = y;
    matrix->giveMem = reallocateMem;
    matrix->freeMem = freeMatrix;
    matrix->inputData = input;
    matrix->data = malloc(sizeof(float)*matrix->x*matrix->y);
    return matrix;
}

