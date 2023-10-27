#ifndef MEMORY_CUH
#define MEMORY_CUH

#include "matrix.h"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

void to_device(Matrix*);

void to_host(Matrix*);

void free_device(Matrix*);

void free_host(Matrix*);

void input_data_h(Matrix*, float*);

void input_data_d(Matrix*, float*);

void allocate_h(Matrix*);

void allocate_d(Matrix*);

void zero(Matrix*);

#endif //MEMORY_CUH