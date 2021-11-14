#ifndef _TENSOR_H
#define _TENSOR_H

#include "shape.h"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

typedef struct Matrix{
    struct Shape shape;
    float *gpuData;
    float *cpuData;
    int gpuAllocation;
    int cpuAllocation;
    void (*copyMemToCpu)(struct Matrix*);
    void (*freeCpu)(struct Matrix*);
    void (*freeGpu)(struct Matrix*);
    void (*flatten)(struct Matrix*);
    void (*inputCpuData)( struct Matrix*, float*);
    void (*inputGpuData)( struct Matrix*, float*);
    void (*allocateCpu)( struct Matrix*, struct Shape);
    void (*allocateGpu)( struct Matrix*, struct Shape);
    void (*freeMem)( struct Matrix*);
    void (*zero)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int, int x, int y, int z);


#endif //TENSOR_H