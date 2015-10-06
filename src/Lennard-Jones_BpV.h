#ifndef __LENNARD_JONES_BPV_H__
#define __LENNARD_JONES_BPV_H__

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
 * Based on testing of the topological distance map
 * With a utilization of only 28%, it appears that the lion share
 * of the time is spent with only one thread, the "first" thread,
 * i.e. thread <- tid % 3, performing work. Reworking to have each
 * thread handle a single atom.
 */
/* shared_array should have enough space for::
 *
 *      DIM*n_atoms*sizeof(float) (scaled atom positions)
 *      n_atoms*sizeof(int) (atom types)
 */
extern __shared__ float shared_array[];
/*
 * Summary
 * -------
 *  Calculates the topological map of the potential energy as
 *  defined by a truncated Lennard-Jones potential.
 *
 * Parameters
 * ----------
 *  :grid (IO): regular grid for the topological map
 *  :n_grid (I): number of points in the grid
 *  :pv (I): periodicity vectors, {xx, xy, xz, yx, yy, yz, zx, zy, zz}
 *           (row-dominant)
 *  :ndiv (I): number of divisions in the grid along each dimension
 *  :scaled_atom_pos (I): scaled atom positions, {x0, y0, z0, x1, ...}
 *      *will be stored in shared array*
 *  :atom_types (I): types of atoms
 *      *will be stored in shared array*
 *  :n_atoms (I): number of atoms
 *  :params (I): LJ parameters: {sigma, epsilon, rcut}
 *      Parameters are stored as {1-1, 1-2, ..., i-j (i <= j), ..., N-N}
 *  :n_types (I): number of types of atoms
 */
__global__ void Lennard_Jones_BpV(\
        volatile float *grid, \
        const int n_grid, \
        const float *pv, \
        const int *ndiv, \
        const float *scaled_atom_pos, \
        const int *atom_types, \
        const int n_atoms, \
        const float **params, \
        const int n_types);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __LENNARD_JONES_BPV_H__ */

