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
    void (*input_data_h)( struct Matrix*, float*, int);
    void (*input_data_d)( struct Matrix*, float*, int);
    void (*allocate_h)( struct Matrix*, struct Shape);
    void (*allocate_d)( struct Matrix*, struct Shape);
    void (*free_matrix)( struct Matrix*);
    void (*zero)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int, int x, int y, int z);


#endif //MATRIX_H