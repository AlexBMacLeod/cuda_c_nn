#ifndef _MATRIX_H
#define _MATRIX_H

#include "shape.h"


typedef struct Matrix{
    struct Shape shape;
    float *memory_d;
    float *memory_h;
    int n;
    void (*to_device)(struct Matrix*);
    void (*to_host)(struct Matrix*);
    void (*free_device)(struct Matrix*);
    void (*free_host)(struct Matrix*);
    void (*flatten)(struct Matrix*);
    void (*input_cpu_data)( struct Matrix*, float*, int);
    void (*input_gpu_data)( struct Matrix*, float*, int);
    void (*allocate_cpu)( struct Matrix*, struct Shape);
    void (*allocate_gpu)( struct Matrix*, struct Shape);
    void (*freeMem)( struct Matrix*);
    void (*zero)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int, int x, int y, int z);


#endif //MATRIX_H