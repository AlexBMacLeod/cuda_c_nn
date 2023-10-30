#ifndef BACKWARD_CUH
#define BACKWARD_CUH

#include "layer.h"


void backward(layer*);

//Needs to be implemented in CUDA
void delta(layer*, float*);

#endif