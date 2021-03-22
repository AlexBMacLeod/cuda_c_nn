//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation_functions.cuh"
#include "Linear.h"
#include "nn.h"
#include "nnKernel.cuh"

#define apply_func(type, func, ...) {
    void *stopper_for_apply = (int[]){0};                               \
    type **list_for_apply = (type*[]){__VA_ARGS__, stopper_for_apply};  \
    for(int i=0; list_for_apply[i] != stopper_for_apply; i++)          \
        fn(list_for_apply[i]);                                          \
        }

#define free_all(...) apply_func(void, free, __VA_ARGS__);


static void freeLayer(struct linearLayer* layer)
{
    free(layer->weights);
    free(layer->derivative);
    free(layer->output);
    free(layer->input);
}

void initLinear(struct linearLayer *layer, int in, int out, struct activation *funcs)
{
    layer->derivative = malloc(sizeof(int)*in*out);
    layer->weights = malloc(sizeof(float)*in*out);
    layer->output = malloc(sizeof(float) * out);
    layer->input = malloc(sizeof (float)*in);
    layer->in = in;
    layer->out = out;
    makeWeights( in, out, layer->weights);
    layer->actFunc = funcs->func;
    layer->derivFunc = funcs->deriv;
    layer->free_layer = freeLayer;
}

float* forward(struct linearLayer* layer, float* input)
{
    memcpy(layer->input, input, sizeof(float)*layer->in);
    matrixVector(layer, input);
    if(layer->actFunc != NULL) layer->actFunc(layer);
    return layer->output;
}

def backward(self, front):
if self.activation:
delta = front*self._relu2deriv(self.out)
out = delta.dot(self.weights.T)
else:
delta = self.out - front
out = delta.dot(self.weights.T)
self.weights -= alpha * self.input.T.dot(delta)
return out
float* backward(struct linearLayer* layer, float* front)
{
    float* delta;
    delta = malloc(sizeof(float)*layer->out);

    if(layer->actFunc != NULL)
    {
        matrixVector(layer, front, delta);
    } else{
        *delta = layer->output - *front;
    }

}