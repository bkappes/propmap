#include "linalg.h"

#ifndef DIM
#define DIM 3
#endif

/*
 * An n-dimensional array can be stored in a 1-D array
 * for efficienct memory access. This function goes
 * from the flattened index, i.e. the index of the
 * 1-D array, to the axis-aligned indices, {i, j, .., N}
 */
__device__ void i_to_ijk(int *ijk, int index, const int *ndiv) {
    int i = 0;
    int stride[DIM];

    stride[0] = 1;
    for (i = 1; i < DIM; ++i) {
        stride[i] = stride[i-1]*ndiv[i-1];
    }
    for (i = DIM; i != 0; --i) {
        ijk[i-1] = index/stride[i-1];
        index %= stride[i-1];
    }
}

