#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#include "layer.h"
#include "activation_functions.cuh"
#include "matrix.h"
#include "nnKernel.cuh"
//#include "../include/nn.h"




void freeLayer(layer* layer)
{
    if(layer->weights!=NULL) layer->weights->free_matrix(layer->weights);
    if(layer->output!=NULL) layer->output->free_matrix(layer->output);
    if(layer->deriv!=NULL) layer->deriv->free_matrix(layer->deriv);
    if(layer->delta!=NULL)layer->delta->free_matrix(layer->delta);
    if(layer->kernels!=NULL)layer->kernels->free_matrix(layer->kernels);
    if(layer->nextWeights==NULL) layer->nextDelta->free_matrix(layer->nextDelta);
    free(layer);
}

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->shape.n; i++)
    {
        for(int j = 0; j < matrix->shape.y; j++)
        {
            matrix->memory_h[i*matrix->shape.y+j] = .2f*(((float)rand()/(float)(RAND_MAX)))-.1f;
        }
    }
}

void forward(layer *layer)
{
    //memset(layer->output->data, 0, layer->out*sizeof(float));
    matrixMultiplication(layer->input, layer->weights, layer->output);
    if(layer->act_func!=NULL)layer->act_func(layer);
    if(layer->deriv_func!=NULL) layer->deriv_func(layer);

}


layer* create_layer(char activation[], LayerType type, struct Shape in, int stride, int in_channels, int out_channels, int kernel_size, bool padding)
{
    layer *layer = malloc(sizeof(layer));

    //layer->flat = in.x*in.y*in.z;

    layer->deriv = NULL;

    layer->batch_size = in.n;
    layer->in = in;
    layer->out = out_channels;
    makeWeights( layer->weights);
    if(type==LAYER_LINEAR){
        layer->weights = createMatrix( in.y, 1, layer->out, 1);
        layer->output = createMatrix( layer->batch_size, 1, layer->out, 1);
        layer->delta = createMatrix( layer->batch_size, 1, in.y, 1);
        
    }else if (type==LAYER_INPUT)
    {
        layer->act_func = NULL;
        layer->deriv_func = NULL;
    }
    
    if(strcmp(activation, "relu") == 0)
    {
        layer->act_func = relu;
        layer->deriv_func = relu_deriv;
        layer->deriv = createMatrix( layer->batch_size, 1, layer->out, 1);
    }else if(strcmp(activation, "softmax")==0){
        layer->act_func = softmax;
        layer->deriv_func = NULL;
    }else if(strcmp(activation, "tanh")==0){
        layer->act_func = tanh;
        layer->deriv_func = tanh_deriv;
        layer->deriv = createMatrix( layer->batch_size, 1, layer->out, 1);
    }else{
        layer->act_func = NULL;
        layer->deriv_func = NULL;
    }

    layer->free_layer = freeLayer;
    layer->forward_pass = forward;
    layer->backward_weights = backward;
    layer->backward_delta = delta;
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;
    return layer;
}


