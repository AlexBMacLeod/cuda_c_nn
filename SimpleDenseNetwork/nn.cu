#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#include "nn.h"

#define ALPHA .1
#define ITERATIONS 150
#define ROWS 4
#define COLS 3
#define OUT 1

typedef struct layer {
    float *weights;
    float *neurons;
    int input;
    int output;
} layer;

void createLayers(layer *hiddenLayers[], char *argv[])
void makeWeights(int rows, int cols, float *x)

int X[ ] = {1,0,1,
    0,1,1,
    0,0,1,
    1,1,1};
int Y[ ] = {1, 1, 0, 0};

int main( int argc, char *argv[])
{
    layer *hiddenLayers[atoi(argv[1])];
    createLayers(*hiddenLayers, argv[]);
}

void createLayers(layer *hiddenLayers[], char *argv[])
{
    for(int i = 0; i < atoi(argv[1]); i++)
    {
        if(i == 0)
        {
            layer[i].input = COLS;
            layer[i].output = (atoi(argv[1]>i)) ? atoi(argv[i+2]) : OUT;
            makeWeights(layer[i].input, layer[i].output, layer[i].weights);
        } else{
            layer[i].input = layer[i-1].output
            layer[i].output = (atoi(argv[1]>i)) ? atoi(argv[i+2]) : OUT;
            makeWeights(layer[i].input, layer[i].output, layer[i].weights);
        }
    }
}

void makeWeights(int rows, int cols, float *x)
{
    srand(time(NULL));
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            x[i*rows+j] = (((float)rand()/(float)(RAND_MAX)));
        }
    }
}
