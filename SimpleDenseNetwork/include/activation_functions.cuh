//
// Created by alex on 3/21/21.
//

#ifndef LINEAR_ACTIVATION_FUNCTIONS_CUH
#define LINEAR_ACTIVATION_FUNCTIONS_CUH

#include "layer.h"

typedef struct activation{
    void (*func)(struct linearLayer*);
    void (*deriv)(struct linearLayer*);
}activation;

extern struct activation relu;

void relu(layer*);

void relu_deriv(layer*);

void tanh(layer*);

void tanh_deriv(layer*);

void softmax(layer*);



__global__ void 
relu_kernel(float* __restrict__, const float* __restrict__,
                            const unsigned int, const unsigned int);

__global__ void 
relu_deriv_kernel(int* __restrict__, const int* __restrict__,
                                  const unsigned int, const unsigned int);
__global__ void 
tanh_kernel(float* __restrict__,
                            const unsigned int, const unsigned int)

__global__ void 
tanh_deriv_kernel(float* __restrict__, float* __restrict__,
                            const unsigned int, const unsigned int)

__global__ void 
softmax_kernel(float* __restrict__,
                            const unsigned int, const unsigned int,
                            const unsigned int)
#endif //LINEAR_ACTIVATION_FUNCTIONS_CUH
