#ifndef __LINALG_H__
#define __LINALG_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <cuda_runtime.h>
/*
 * Returns the 2-norm of the vector, v[len].
 */
__device__ float v_norm(const float *v, const int len);

/*
 * Evaluates s = x*a + y, where
 *   s = 1 x acols (row vector)
 *   x = 1 x xcols (row vector)
 *   a = xcols x acols (matrix, row-dominant -- i.e. C-style)
 *   y = 1 x acols (row vector, or NULL)
 */
__device__ void vm_saxpy(float *s, \
        const float *x, \
        const int xcols, \
        const float *a, \
        const int acols, \
        const float *y);

/*
 * Converts the flattened index, i, into the axis-aligned
 * index, ijk, assuming axis 0 varies most quickly, then
 * axis 1, then axis 2, and so forth.
 *   ndiv = number of divisions along each dimension.
 */
__device__ void i_to_ijk(int *ijk, int i, const int *ndiv);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __LINALG_H__ */

