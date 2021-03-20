#pragma once

float MSE(float yhat, float y);

float * makeWeights(int rows, int cols);

void printNetwork(layer *hiddenLayers[], char *argv[]);
