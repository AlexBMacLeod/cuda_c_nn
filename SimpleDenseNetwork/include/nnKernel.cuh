__global__ void vecxvec_kernel(const float* __restrict__ d_m, const float* __restrict__ d_x, float * __restrict__ d_p,
    const unsigned int nRows, const unsigned int nCols)

__global__ void matvec_kernel(const float* __restrict__ d_M, const float* __restrict__ d_x, float * __restrict__ d_p, 
    const unsigned int nRows, const unsigned int nCols, bool ReLU)

void forwardPass(float *inputRow, layer hiddenLayers[],
    float *yhat, int numHidden)

void backpass(layer *hiddenLayers[], float *yhat, float alpha)