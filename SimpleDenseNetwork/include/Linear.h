//
// Created by alex on 3/21/21.
//


#ifndef LINEAR_LINEAR_H
#define LINEAR_LINEAR_H

typedef struct linearLayer{
    float *weights;
    int *derivative;
    float *output;
    float *input;
    int in;
    int out;
    void (*actFunc)(struct linearLayer*);
    void (*derivFunc)(struct linearLayer*)
    void (*forward_pass)(struct linearLayer*);
    void (*backward_pass)(struct linearLayer*);
    void (*free_layer)(struct linearLayer*);
};

static void freeLayer(struct linearLayer*);

void initLinear(struct linearLayer* layer, int rows, int cols, struct activation *funcs);

float* forward(struct linearLayer*, float*);
float* backward(struct linearLayer*, float*);

#endif //LINEAR_LINEAR_H
