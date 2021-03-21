//
// Created by alex on 3/21/21.
//

#ifndef LINEAR_ACTIVATION_FUNCTIONS_CUH
#define LINEAR_ACTIVATION_FUNCTIONS_CUH

typedef struct activation{
    void (*func)(struct linearLayer*);
    void (*deriv)(struct linearLayer*);
};

void relu_func(struct linearLayer*);
void relu_deriv(struct linearLayer*);

__global__ void relu_kernel(float* __restrict__ d_m, const float* __restrict__ d_x,
                            const unsigned int nRows, const unsigned int nCols);

__global__ void relu_deriv_kernel(int* __restrict__ d_out, const int* __restrict__ d_in,
                                  const unsigned int nRows, const unsigned int nCols);

#endif //LINEAR_ACTIVATION_FUNCTIONS_CUH
