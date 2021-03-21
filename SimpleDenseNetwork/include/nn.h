#pragma once

float MSE(float yhat, float y);

void makeWeights(int rows, int cols, float *x);

void printNetwork(layer *hiddenLayers[], char *argv[]);


