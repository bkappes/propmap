#ifndef __DISTANCE_BPV_H__
#define __DISTANCE_BPV_H__

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


#ifndef DIM
#define DIM 3
#endif
/*
 * ==================================================
 * TpD (thread per direction)
 * ==================================================
 * Calculate the distance with each voxel handled by one (or more)
 * blocks. The number of threads per block is device dependent, but
 * my current card can handle 1024 threads per block. If each thread
 * handles a single dimension (x, y, z) for an atom, this equals
 * 1024/3 = 341 atoms per block. Therefore, if there are more than
 * 341 atoms in my crystal, then I must use more than one block for
 * each voxel. What follows is the grid and block structure for this
 * code:
 *
 *   block = (max_threads_per_block, 1, 1)
 *   grid = (num_voxels, 1 + 3*num_atoms/max_threads_per_block, 1)
 * ==================================================
 * TpA (thread per atom)
 * ==================================================
 * With a utilization of only 28%, it appears that the lion share
 * of the time is spent with only one thread, the "first" thread,
 * i.e. thread <- tid % 3, performing work. Reworking to have each
 * thread handle a single atom.
 */
// shared_array should have enough space for the DIM*n_atoms floats
// to which distances are calculated
extern __shared__ float shared_array[];
__global__ void distance_BpV(\
        volatile float *grid, \
        const int n_grid, \
        const float *pv, \
        const int *ndiv, \
        const float *scaled_atom_pos, \
        const float *atomic_radii, \
        const int n_atoms);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __DISTANCE_BPV_H__ */

