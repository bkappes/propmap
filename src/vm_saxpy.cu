#include "linalg.h"
#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

__device__ void vm_saxpy(float * s, \
        const float * x, \
        const int xcols, \
        const float * a, \
        const int acols, \
        const float * y) {
    int i = 0;
    int j = 0;
    assert(s != x);
    for (j = 0; j < acols; ++j) {
        s[j] = (y != NULL) ? y[j] : 0.0;
        for (i = 0; i < xcols; ++i) {
            s[j] += x[i]*a[i*acols + j];
        }
    }
}

