
#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H


typedef struct Matrix{
    union{
        struct{ int x, y;
        };
    };
    float *data;
    void (*inputData)( struct Matrix*, float*);
    void (*giveMem)( struct Matrix*, int, int);
    void (*freeMem)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int, int);
static void input(Matrix*, float*);
static void reallocateMem( Matrix*, int, int);
static void freeMatrix( Matrix*);

#endif