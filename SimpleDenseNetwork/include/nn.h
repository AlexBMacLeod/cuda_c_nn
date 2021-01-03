#include <stdio.h>
#include <stdlib.h>

float MSE(float yhat, float y)
{
    float mse = (y - yhat)**2;
    return mse;
}

float * makeWeights(int rows, int cols)
{
    float x[rows*cols];
    srand(time(NULL));
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            x[i*rows+j] = (((float)rand()/(float)(RAND_MAX)));
        }
    }
    return x;
}

void printNetwork(layer *hiddenLayers[], char *argv[])
{
    printf("Layer 0:\nInput %d\n", COLS);
    for(int i = 0, i < atoi(argv[1]); i++)
    {
        printf("Layer %d:\nInput %d\tOutput %d\n");
    }
    printf("Output Layer:\nOutput %d", OUTPUT);
}
