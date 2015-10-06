#include "Lennard_Jones_BpV.h"
#include "linalg.h"
#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

/*
 * Summary
 * -------
 *  Returns the potential energy from a truncated 12-6 Lennard-Jones
 *  force field. To avoid overflow errors, this takes the form:
 *
 *            / U(sigma/1.e-3) for sigma/r < 1.e-3
 *    U(r) = |  4 * epsilon * ((sigma/r)^12 - (sigma/r)^6) for r < rcut
 *            \ 0.0 otherwise
 *
 * Parameters
 * ----------
 *   :sigma (I): sigma parameter for LJ
 *   :epsilon (I): epsilon parameter for LJ
 *   :rcut (I): truncation cutoff radius
 *   :r (I): radius at which the energy is to be calculated
 */
__device__ float truncated_12_6_LJ(const float sigma, \
        const float epsilon, \
        const float rcut, \
        const float r) {
    assert(rcut > 0.0);
    assert(r > 0.0);
    float sr_ratio = 1.0f; // sigma-r ratio
    float sc_ratio = sigma/rcut; // sigma-rcut ratio
    float retval = 0.0f;
    if (sigma/r < 1.e-3f) {
        retval = truncated_12_6_LJ(sigma/1.e-3f);
    } else if (r < rcut) {
        sr_ratio = sigma/r;
        // squared
        sr_ratio *= sr_ratio;
        sc_ratio *= sc_ratio;
        // sixth power
        sr_ratio *= sr_ratio * sr_ratio;
        sc_ratio *= sc_ratio * sc_ratio;
        // LJ
        retval = 4.0f * epsilon * (sr_ratio*sr_ratio - sr_ratio) - \
                 4.0f * epsilon * (sc_ratio*sc_ratio - sc_ratio);
    } else {
        retval = 0.0;
    }
    return retval;
}

/*
 * Synopsis
 * --------
 * Calculates the Lennard-Jones energy of a voxel interacting with
 * each atom in the structure. The interaction with each atom is
 * calculated only for the nearest image, that is, the unit cell
 * is assumed to satisfy the minimum image criterion.
 *
 * Args
 * ----
 *   :grid (out): flattened regular grid to hold the energy. The
 *       contribution from each atom will be added to the value
 *       stored in the grid when the function is called, so this
 *       should generally be initialized with zeroes.
 *
 *   :n_grid (in): size of (number of voxels in) the grid
 *
 *   :pv (in): periodicity vectors in C-style storage
 *
 *   :ndiv (in): number of divisions along each dimension, i.e.
 *       how many partitions (bins) per axis
 *
 *   :scaled_atom_pos (in): atom positions in fractional coordinates
 *
 *   :atom_types (in): the type (index) of each atom corresponding
 *       to its location in the parameterization, *params*, matrix
 *
 *   :n_atoms (in): the number of atoms in the system
 *
 *   :params (in): parameterization of the potential
 * 
 *       .. code::
 * 
 *           params = [[sigma_{voxel, 0}, sigma_{voxel, 1}, ..., sigma_{voxel, N}],
 *                     [epsilon_{voxel, 0}, epsilon_{voxel, 1}, ..., epsilon_{voxel, N}],
 *                     [rcut_{voxel, 0}, rcut_{voxel, 1}, ..., rcut_{voxel, N}]]
 *
 *   :n_types (in): the number of types in the potential parameterization.
 *       This precisely equals *N* in the code block above (in the
 *       description of *params*).
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
        const int n_types) {
    // One (or more) blocks handle a voxel, e.g. Blocks per Voxel (BpV)
    // distance between an atom and a voxel
    const int vid = blockIdx.x; // voxel index
    const int bid = blockIdx.y; // block index
    // atom index
    const int aid = threadIdx.x + \
                    blockDim.x * threadIdx.y + \
                    blockDim.x * blockDim.y * threadIdx.z;
    const int BLOCK_SIZE = blockDim.x * blockDim.y * blockDim.z;
    const int MAX_ATOMS = BLOCK_SIZE;
    const int ATOMS_THIS_BLOCK = max(min(MAX_ATOMS, \
                                         n_atoms - MAX_ATOMS*bid), \
                                     0);
    /* **************************************************
     * shared array variables
     * **************************************************/
    // atom positions
    float *dx = shared_array;
    /* **************************************************
     * shared array size
     * -----------------
     *   DIM*n_atoms * sizeof(float)
     * **************************************************/

    // parameterization of the potential
    //... voxel/type_0 interaction, voxel/type_1 interaction, etc.
    const float *sigma = params[0];
    const float *epsilon = params[1];
    const float *rcut = params[2];

    // local variables
    int i = 0, j = 0;
    int gaid = 0; // global atom ID
    int ijk[DIM];
    float voxel_scaled = 0.0f;
    float dst[DIM];

    // flattened grid index, vid, to axis-aligned indices, ijk
    i_to_ijk(ijk, vid, ndiv);

    // calculate the energy between each voxel and each atom
    if (aid < ATOMS_THIS_BLOCK) {
        gaid = aid + bid*MAX_ATOMS;
        // component distance, in scaled coordinates, between [-0.5, 0.5)
        for (i = 0; i < DIM; ++i) {
            j = DIM*aid + i;
            dx[j] = scaled_atom_pos[DIM*gaid + i];
            // scaled voxel coordinate
            voxel_scaled = (float)(ijk[i])/(float)(ndiv[i]);
            // distance from this atom dimesion to the
            // corresponding voxel dimension
            dx[j] = voxel_scaled - dx[j];
            // roundup, i.e. ensure all distances lie in [-0.5, 0.5)
            // (scaled coordinate)
            dx[j] -= floor(dx[j] + 0.5f);
            assert(dx[j] >= -0.5f && dx[j] < 0.5f);
        }
        // convert component distance from scaled to real coordinates
        vm_saxpy(dst,
                 &dx[DIM*gaid], DIM,
                 pv, DIM, NULL);
        // find the energy contribution from each atom
        // and store in the first index of that atom
        // At the end of this loop:
        // dx = {E0, y0, z0, E1, y1, z1, ..., EN, yN, zN}
        // note: E, not x. The x-coordinate has been lost.
        dx[DIM*gaid] = truncated_12_6_LJ(sigma[gaid], \
                epsilon[gaid], \
                rcut[gaid], \
                v_norm(dst, DIM));
    }

    __syncthreads();

    /* reduce algorithm to find the sum of the energies */
    // put all radii at the beginning
    if (ATOMS_THIS_BLOCK > 1) {
        /*
         * Why not use a bit shift to divide by 2?
         * If 2 atoms, then atom 1 <- op(atom 1, atom 2) and done.
         * If 3 atoms, atom 1 <- op(atom 1, atom 3), atom 2 unchanged,
         * <<step 2>> atom 1 <- op(atom 1, atom 2), done.
         * If 4 atoms, atom 1 <- op(atom 1, atom 3) and
         * atom 2 <- op(atom 2, atom 4), <<step 2>>
         * atom1 <- op(atom 1, atom 2), done.
         * If 5 atoms, atom 1 <- op(atom 1, atom 4), atom 2 <- op(atom 2, atom 5)
         * keep atom 3, <<step 2>> atom 1 <- op(atom 1, atom 3), keep atom 2,
         * <<step 3>> atom 1 <- op(atom 1, atom 2), done.
         * If N atoms, atom 1 <- op(atom 1, atom (N+1)/2), ...
         * which leaves the lower (N+1)/2 atoms modified and the upper
         * N - (N+1)/2 junk. Now a scenario of "If (N+1)/2 atoms, atom 1..."
         */
        // set the upper limit (exclusive) for the atom to examine
        if (aid == 0)
            j = ATOMS_THIS_BLOCK;
        __syncthreads();
        for (i = (ATOMS_THIS_BLOCK + 1)/2; i != 1; i = (i + 1)/2) {
            // in case odd, do not try to access
            // one-past-the-end.
            if ((aid < i) && (aid+i < j))
                dx[DIM*aid] += dx[DIM*(aid+i)];
            __syncthreads();
            if (aid == 0)
                j = i;
            __syncthreads();
        }
        // previous loop always completes with 2 atoms remaining
        if (aid == 0)
            dx[0] += dx[DIM];
    }

    // add the value
    if (aid == 0 && vid < n_grid)
        atomicAdd(&grid[vid], dx[0]);

    __syncthreads();
}

#include "i_to_ijk.cu"
#include "vm_saxpy.cu"
#include "v_norm.cu"

