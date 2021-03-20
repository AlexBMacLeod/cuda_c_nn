#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/nnKernel.cuh"
#include "../include/nn.h"

#define ALPHA .1
#define ITERATIONS 150
#define ROWS 4
#define COLS 3
#define OUT 1

void createLayers(layer *hiddenLayers[], char *argv[])

int X[ ] = {1,0,1,
    0,1,1,
    0,0,1,
    1,1,1};
int Y[ ] = {1, 1, 0, 0};

int main( int argc, char *argv[])
{
    layer *hiddenLayers[atoi(argv[1])];
    createLayers(*hiddenLayers, argv[]);
    printNetwork()
}

void createLayers(layer *hiddenLayers[], char *argv[])
{
    for(int i = 0; i < atoi(argv[1]); i++)
            layer[i].input = {
                .input = (i==0)COLS:layer[i-1].output, 
                .output =  (atoi(argv[1]!=i+1)) ? atoi(argv[i+2]) : OUT,
                .weights = makeWeights(layer[i].input, layer[i].output)}
        }
    }
}