#include "linalg.h"

/*
 * Calculates the vector 2-norm.
 * -----------------------------
 *
 * Parameters
 * ----------
 * :v (float*): vector
 * :len (int): length (dimension) of the vector
 */
__device__ float v_norm(const float *v, const int len) {
    int i = 0;
    float rval = 0.0;
    for (i = 0; i < len; ++i) {
        rval += v[i]*v[i];
    }
    return sqrtf(rval);
}

