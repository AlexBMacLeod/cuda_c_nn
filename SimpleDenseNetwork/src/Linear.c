//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <stdlib.h>

#include "activation_functions.cuh"
#include "Linear.h"
#include "nn.h"




static void freeLayer(struct linearLayer* layer)
{
    free(layer->weights);
    free(layer->derivative);
    free(layer->output);
}

void initLinear(struct linearLayer *layer, int in, int out, struct activation *funcs)
{
    layer->derivative = malloc(sizeof(int)*in*out);
    layer->weights = malloc(sizeof(float)*in*out);
    layer->output = malloc(sizeof(float) * out);
    makeWeights( in, out, layer->weights);
    layer->acvitation_function = funcs;
    layer->free_layer = freeLayer;
}