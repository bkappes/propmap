#include "distance_BpV.h"
#include "linalg.h"
#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

__global__ void distance_BpV(\
        volatile float *grid, \
        const int n_grid, \
        const float *pv, \
        const int *ndiv, \
        const float *scaled_atom_pos, \
        const float *atomic_radii, \
        const int n_atoms) {
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
    const int ATOMS_THIS_BLOCK = max(min(MAX_ATOMS, n_atoms - MAX_ATOMS*bid), 0);

    // shared array variables
    // atom positions
    float *dx = shared_array;

    int i = 0, j = 0;
    int ijk[DIM];
    float voxel_scaled = 0.0;
    float dst[DIM];

    // flattened grid index, vid, to axis-aligned indices, ijk
    i_to_ijk(ijk, vid, ndiv);

    // round up the atom distance calculation
    if (aid < ATOMS_THIS_BLOCK) {
        // component distance, in scaled coordinates, between [-0.5, 0.5)
        for (i = 0; i < DIM; ++i) {
            j = DIM*aid + i;
            dx[j] = scaled_atom_pos[DIM*(aid + bid*MAX_ATOMS) + i];
            // scaled voxel coordinate
            voxel_scaled = (float)(ijk[i])/(float)(ndiv[i]);
            // distance from this atom dimesion to the corresponding voxel dimension
            dx[j] = voxel_scaled - dx[j];
            // roundup, i.e. ensure all distances lie in [-0.5, 0.5)
            // (scaled coordinate)
            dx[j] -= floor(dx[j] + 0.5);
            assert(dx[j] >= -0.5 && dx[j] < 0.5);
        }
        // convert component distance from scaled to real coordinates
        vm_saxpy(dst,
                 &dx[DIM*aid], DIM,
                 pv, DIM, NULL);
        // find the distances to each atom
        // and store in the first index of each
        // atom. At the end of this loop:
        // dx = {r0, y0, z0, r1, y1, z1, ..., rN, yN, zN}
        // note: r, not x. The x-coordinate has been lost.
        dx[DIM*aid] = v_norm(dst, DIM) - atomic_radii[aid + bid*MAX_ATOMS];
    }

    __syncthreads();

    /* reduce algorithm to find the minimum dr */
    // put all radii at the beginning
    if (ATOMS_THIS_BLOCK > 1) {
        /*
         * Why not use a bit shift to divide by 2?
         * If 2 atoms, then atom 1 <- min(atom 1, atom 2) and done.
         * If 3 atoms, atom 1 <- min(atom 1, atom 3), atom 2 unchanged,
         * <<step 2>> atom 1 <- min(atom 1, atom 2), done.
         * If 4 atoms, atom 1 <- min(atom 1, atom 3) and
         * atom 2 <- min(atom 2, atom 4), <<step 2>>
         * atom1 <- min(atom 1, atom 2), done.
         * If 5 atoms, atom 1 <- min(atom 1, atom 4), atom 2 <- min(atom 2, atom 5)
         * keep atom 3, <<step 2>> atom 1 <- min(atom 1, atom 3), keep atom 2,
         * <<step 3>> atom 1 <- min(atom 1, atom 2), done.
         * If N atoms, atom 1 <- min(atom 1, atom (N+1)/2), ...
         * which leaves the lower (N+1)/2 atoms modified and the upper
         * N - (N+1)/2 junk. Now a scenario of "If (N+1)/2 atoms, atom 1..."
         */
        for (i = (ATOMS_THIS_BLOCK + 1)/2; i != 1; i = (i + 1)/2) {
            // in case odd, do not try to access
            // one-past-the-end. Because distances
            // are stored in every DIM indices, sort
            // considers only these DIM*i indices.
            if ((aid < i) && (aid+i < ATOMS_THIS_BLOCK))
                dx[DIM*aid] = min(dx[DIM*aid], dx[DIM*(aid+i)]);
            __syncthreads();
        }
        // for an odd number of atoms, ensure the "kept" atom is
        // compared
        if (aid == 0)
            dx[0] = min(dx[0], dx[DIM]);
        __syncthreads();
    }

    // Store the minimum distance
    /* This should probably be atomic, but the chance that multiple blocks will reach this
     * point at the same time is very, very unlikely. */
    if (aid == 0 && vid < n_grid)
        grid[vid] = min(grid[vid], dx[0]);

    __syncthreads();
}

#include "i_to_ijk.cu"
#include "vm_saxpy.cu"
#include "v_norm.cu"

