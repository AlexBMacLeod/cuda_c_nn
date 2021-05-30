
#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "shape.h"

typedef struct Matrix{
    struct Shape shape;
    float *data;
    void (*inputData)( struct Matrix*, float*);
    void (*giveMem)( struct Matrix*, struct Shape);
    void (*freeMem)( struct Matrix*);
}Matrix;

Matrix* createMatrix( struct Shape);
static void input(Matrix*, float*);
static void reallocateMem( Matrix*, struct Shape);
static void freeMatrix( Matrix*);

#endif