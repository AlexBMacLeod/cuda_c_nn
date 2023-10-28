#ifndef _LAYER_H
#define _LAYER_H

#include <stdbool.h>

#include "matrix.h"


//Previously I had a different kind of layer for each layer type
//But managing them all was problematic so I made one generic layer which can be customized for each layer type using a mix of 
//pointers, unions and shared properties.

typedef enum {
    LAYER_LINEAR,
    LAYER_CONV2D,
    LAYER_INPUT
} LayerType;

typedef struct layer{
    LayerType layer_type;
    Matrix *output;
    Matrix *deriv;
    Matrix *input;
    Matrix *delta;
    Matrix *nextDelta;
    Matrix *nextWeights;
    Matrix *weights;
    Matrix *kernels;
    float lr;
    int batch_size;
    bool padding;
    int batch_size;
    int stride;
    int flat;
    struct Shape in;
    int out;
    void (*act_func)(struct layer*);
    void (*deriv_func)(struct layer*);
    void (*forward_pass)(struct layer*);
    void (*backward_weights)(struct layer*);
    void (*backward_delta)(struct layer*, float*);
    void (*free_layer)(struct layer*);
} layer;

void makeWeights( Matrix*);

void free_layer(layer*);

//void initLinear( layer*, int, int, activation *funcs);
void init_layer(layer**, char[], int, int);

layer* create_layer(char[], LayerType, struct Shape, int, int, int, int, bool);

void forward(layer*);
//
void backward(layer*);

void delta(layer*, float*);

void weight_update(layer*);


#endif