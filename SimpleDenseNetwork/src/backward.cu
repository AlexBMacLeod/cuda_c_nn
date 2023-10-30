#include <cuda.h>
#include <cuda_runtime.h>

#include "backward.cuh"
#include "layer.h"


//Needs to be implemented in CUDA
void backward(layer* layer)
{
    Matrix *invInput = createInverse(layer->input);
    Matrix *weightsDelta = createMatrix(layer->in.y, 1, layer->out, 1);
    matrixMultiplication(invInput, layer->nextDelta, weightsDelta);
    matrixScalarMultiplicationInPlace(weightsDelta, layer->lr);
    matrixSubtraction(layer->weights, weightsDelta);
    weightsDelta->free_matrix(weightsDelta);
    invInput->free_matrix(invInput);
}

//Needs to be implemented in CUDA
void delta(layer* layer, float* y)
{
    if(layer->nextWeights==NULL)
    {   
        if(layer->nextDelta==NULL) layer->nextDelta = createMatrix(layer->batch_size, 1 , layer->out, 1);
        float b_size = layer->batch_size;
        for(int i=0; i<layer->nextDelta->shape.n; i++)
        {
            for(int j=0; j<layer->nextDelta->shape.y; j++)
            {
                layer->nextDelta->data[i*layer->out+j] = (layer->output->data[i*layer->out+j] - y[i*layer->out+j])/b_size;
            }
        }
    }
    if(layer->deriv!=NULL) elemMatrixMultInPlace(layer->nextDelta, layer->deriv);
    Matrix *invWeights = createInverse(layer->weights);
    matrixMultiplication(layer->nextDelta, invWeights, layer->delta);
    invWeights->free_matrix(invWeights);
}