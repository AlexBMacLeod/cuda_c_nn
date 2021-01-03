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
