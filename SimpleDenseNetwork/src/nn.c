#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "../include/nn.h"


/*float MSE(float yhat, float y)
{
    float mse = pow((y - yhat), 2.0);
    return mse;
}*/

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->y; i++)
    {
        for(int j = 0; j < matrix->x; j++)
        {
            matrix->data[i*matrix->x+j] = (((float)rand()/(float)(RAND_MAX)));
        }
    }
}

/*void printNetwork(layer *hiddenLayers[], char *argv[])
{
    printf("Layer 0:\nInput %d\n", COLS);
    for(int i = 0, i < atoi(argv[1]); i++)
    {
        printf("Layer %d:\nInput %d\tOutput %d\n");
    }
    printf("Output Layer:\nOutput %d", OUTPUT);
}
*/