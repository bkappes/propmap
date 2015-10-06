#include "distance_BpV.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// assert() is only supported for devices of compute capability 2.0 and higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif


// for easy allocation/deallocation
typedef struct __dynamic_ptrs {
    void **__host;
    void **__device;
} dynamic_ptrs;

// declarations
void* new_host(dynamic_ptrs*, size_t);
void* new_device(dynamic_ptrs*, size_t);
void cleanup(dynamic_ptrs*);
void construct(dynamic_ptrs**);
void destroy(dynamic_ptrs**);
void check_cuda(int id, dynamic_ptrs *ptrs);
void failed(const char *msg);
void passed(const char *msg);
void started(const char *msg);

// new object on the host
void* new_host(dynamic_ptrs *self, size_t nbytes) {
    void *ptr = malloc(nbytes);
    if (self && ptr) {
        // store this pointer in a list
        if (self->__host) {
            size_t nptrs = 0;
            char **iit = NULL;
            char **dst = NULL;
            // how many pointers
            nptrs = 0;
            for (iit = (char**) self->__host; *iit != NULL; ++iit)
                ++nptrs;
            // copy pointers
            dst = (char**) malloc((nptrs+2)*sizeof(char*));
            memcpy(dst, self->__host, nptrs*sizeof(void*));
            free(self->__host);
            // add new pointer
            dst[nptrs++] = (char*) ptr;
            // NULL terminator
            dst[nptrs++] = NULL;
            // copy back
            self->__host = (void**) malloc(nptrs*sizeof(void*));
            memcpy(self->__host, dst, nptrs*sizeof(void*));
            free(dst);
        } else {
            char **iit = NULL;
            self->__host = (void**) malloc(2*sizeof(void*));
            iit = (char**) self->__host;
            iit[0] = (char*) ptr;
            iit[1] = NULL;
        }
    }
    return ptr;
}

// new object on the device
void* new_device(dynamic_ptrs *self, size_t nbytes) {
    void *ptr = NULL;
    cudaError_t rval;
    rval = cudaMalloc(&ptr, nbytes);
    if (rval != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(rval));
        printf("          : Failed to allocate %lu bytes, e.g. %.2f floats or %.2f ints\n", \
               nbytes, \
               (float) nbytes/sizeof(float), \
               (float) nbytes/sizeof(int));
        printf("BAILING OUT!\n");
        cleanup(self);
        exit(1);
    }
    if (self && ptr) {
        // store this pointer in a list
        if (self->__device) {
            size_t nptrs = 0;
            char **iit = NULL;
            char **dst = NULL;
            // how many pointers
            nptrs = 0;
            for (iit = (char**) self->__device; *iit != NULL; ++iit)
                ++nptrs;
            // copy pointers
            dst = (char**) malloc((nptrs+2)*sizeof(char*));
            memcpy(dst, self->__device, nptrs*sizeof(void*));
            free(self->__device);
            // add new pointer
            dst[nptrs++] = (char*) ptr;
            // NULL terminator
            dst[nptrs++] = NULL;
            // copy back
            self->__device = (void**) malloc(nptrs*sizeof(void*));
            memcpy(self->__device, dst, nptrs*sizeof(void*));
            free(dst);
        } else {
            char **iit = NULL;
            self->__device = (void**) malloc(2*sizeof(void*));
            iit = (char**) self->__device;
            iit[0] = (char*) ptr;
            iit[1] = NULL;
        }
    }
    return ptr;
}

// cleanup function
void cleanup(dynamic_ptrs *self) {
    if (self->__host) {
        char **ptr = (char**) self->__host;
        for (; *ptr != NULL; ++ptr)
            free(*ptr);
        free(self->__host);
        self->__host = NULL;
    }
    if (self->__device) {
        char **ptr = (char**) self->__device;
        for (; *ptr != NULL; ++ptr)
            cudaFree(*ptr);
        free(self->__device);
        self->__device = NULL;
    }
}

// construct a new structure of dynamic pointers
void construct(dynamic_ptrs **dst) {
    *dst = (dynamic_ptrs*) malloc(sizeof(dynamic_ptrs));
    (*dst)->__host = NULL;
    (*dst)->__device = NULL;
}

// destroy a new structure of dynamic pointers
void destroy(dynamic_ptrs **dst) {
    if (dst && *dst) {
        cleanup(*dst);
        free(*dst);
        *dst = NULL;
    }
}

void check_cuda(int id, dynamic_ptrs *ptrs) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%d. CUDA error: %s\n", id, cudaGetErrorString(error));
        if (ptrs)
            cleanup(ptrs);
        exit(1);
    }
}

void failed(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 13; ++i)
        printf("-");
    printf("\n");
    printf("TEST FAILED: %s\n", msg);
    printf("\n");
}

void passed(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 13; ++i)
        printf("-");
    printf("\n");
    printf("TEST PASSED: %s\n", msg);
    printf("\n");
}

void started(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 14; ++i)
        printf("=");
    printf("\n");
    printf("STARTED TEST: %s\n", msg);
    for (i = 0 ; i < strlen(msg) + 14; ++i)
        printf("-");
    printf("\n");
}

__device__ float dfloat;

int test_v_norm();
int test_vm_saxpy();
int test_i_to_ijk();
int test_distance_BpV();
__global__ void call_v_norm(const float *v, const int len) {
    dfloat = v_norm(v, len);
}
__global__ void call_vm_saxpy(float *s, const float *a, const int acols, const float *x, const int xcols, const float *y) {
    vm_saxpy(s, a, acols, x, xcols, y);
}
__global__ void call_i_to_ijk(int *ijk, int i, const int *ndiv) {
    i_to_ijk(ijk, i, ndiv);
}

int main(int argc, const char *argv[]) {
    int i = 0;
    int nDevices = 0;
    struct cudaDeviceProp deviceProps;

    started("System Info");

    // information for device(s)
    cudaGetDeviceCount(&nDevices);
    for (i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
    }
    cudaSetDevice(i-1); // select the last device
    check_cuda(1, NULL);

    // tests
    if (test_v_norm()) failed("v_norm");
    else passed("v_norm");

    if(test_vm_saxpy()) failed("vm_saxpy");
    else passed("vm_saxpy");

    if(test_i_to_ijk()) failed("i_to_ijk");
    else passed("i_to_ijk");

    if(test_distance_BpV()) failed("distance_BpV");
    else passed("distance_BpV");
}

int test_v_norm() {
    int i = 0;
    int nelem = 8;
    int nbytes = nelem*sizeof(float);
    int rval = 1;
    float hsol = 0.0;
    float dsol = 0.0;
    float *hvec = NULL;
    float *dvec = NULL;

    started("v_norm");

    srandom(time(NULL));

    // host vector
    hvec = (float*) malloc(nbytes);
    for (i = 0; i < nelem; ++i)
        hvec[i] = (float)(random())/RAND_MAX;

    // host solution
    hsol = 0.0;
    for (i = 0; i < nelem; ++i)
        hsol += hvec[i]*hvec[i];
    hsol = sqrt(hsol);

    // device vector
    cudaMalloc((void**) &dvec, nbytes); check_cuda(1, NULL);
    cudaMemcpy(dvec, hvec, nbytes, cudaMemcpyHostToDevice); check_cuda(2, NULL);
    call_v_norm<<<1, 1>>>(dvec, nelem); check_cuda(3, NULL);
    cudaMemcpyFromSymbol(&dsol, dfloat, sizeof(dsol)); check_cuda(4, NULL);

    // check the results
    printf("  |v(%d)|: ", nelem);
    dsol = fabs(1.0 - dsol/hsol);
    if (dsol > 1.e-6) {
        printf("failed, (cpu, gpu) = (%f, %f)\n", hsol, dsol);
        rval = 1;
    } else {
        printf("passed\n");
        rval = 0;
    }

    // cleanup
    free(hvec);
    cudaFree(dvec); check_cuda(5, NULL);

    return rval;
}

int test_vm_saxpy() {
    int i = 0;
    int j = 0;
    const int m = 3, n = 5; // 1xm * mxn = 1xn
    // all vectors/matrices in row dominant (C-style) format
    float *soln = NULL;
    float *hs = NULL;
    float *ha = NULL;
    float *hx = NULL;
    float *hy = NULL;
    float *ds = NULL;
    float *da = NULL;
    float *dx = NULL;
    float *dy = NULL;
    // booleans
    int check = 0;
    int rval = 0;

    started("vm_saxpy");

    // seed the random number generator
    srandom(time(NULL));

    // allocate memory
    soln = (float*) malloc(n*sizeof(float));
    hs = (float*) malloc(n*sizeof(float));
    cudaMalloc(&ds, n*sizeof(float));
    hx = (float*) malloc(m*sizeof(float));
    cudaMalloc(&dx, m*sizeof(float));
    ha = (float*) malloc(m*n*sizeof(float));
    cudaMalloc(&da, m*n*sizeof(float));
    hy = (float*) malloc(n*sizeof(float));
    cudaMalloc(&dy, n*sizeof(float));

    // generate host data
    memset(hs, 0, n*sizeof(float));
    for (i = 0; i < m; ++i)
        hx[i] = (float)(random())/RAND_MAX;
    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j)
            ha[i*n + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < n; ++i)
        hy[i] = (float)(random())/RAND_MAX;
    // host solution, y = NULL
    for (j = 0; j < n; ++j)
        for (i = 0; i < m; ++i)
            hs[j] += hx[i]*ha[i*n + j];

    // device solution, y = NULL
    cudaMemcpy(dx, hx, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(da, ha, m*n*sizeof(float), cudaMemcpyHostToDevice);
    call_vm_saxpy<<<1, 1>>>(ds, dx, m, da, n, NULL);
    cudaMemcpy(soln, ds, n*sizeof(float), cudaMemcpyDeviceToHost);

    // check
    rval = 0;
    // check for y = NULL
    check = 0;
    for (i = 0; i < n; ++i)
        if(fabs(1.0 - hs[i]/soln[i]) > 1.e-6)
            check = 1;
    printf("  s = x*a: ");
    if (check) {
        printf("failed\n");
        printf("  >>> ( ");
        for (i = 0; i < n; ++i)
            printf("%f ", hs[i]);
        printf(") = ( ");
        for (i = 0; i < n; ++i)
            printf("%f ", soln[i]);
        printf(")\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // host solution, y != NULL
    for (i = 0; i < n; ++i)
        hs[i] += hy[i];

    // device solution y != NULL
    cudaMemcpy(dy, hy, n*sizeof(float), cudaMemcpyHostToDevice);
    call_vm_saxpy<<<1, 1>>>(ds, dx, m, da, n, dy);
    cudaMemcpy(soln, ds, n*sizeof(float), cudaMemcpyDeviceToHost);

    // check for y != NULL
    check = 0;
    for (i = 0; i < n; ++i)
        if (fabs(1.0 - hs[i]/soln[i]) > 1.e-6)
            check = 1;
    printf("  s = x*a + y: ");
    if (check) {
        printf("failed\n");
        printf("  >>> ( ");
        for (i = 0; i < n; ++i)
            printf("%f ", hs[i]);
        printf(") = ( ");
        for (i = 0; i < n; ++i)
            printf("%f ", soln[i]);
        printf(")\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // cleanup
    free(soln);
    free(hs);
    free(ha);
    free(hx);
    free(hy);
    cudaFree(ds);
    cudaFree(da);
    cudaFree(dx);
    cudaFree(dy);

    return rval;
}

int test_i_to_ijk() {
    int i;
    const int dim = DIM;
    const int indices[] = {0, 46, 4562, 84328, -1};
    const int *it = NULL;
    int index = 0;
    int *soln = NULL;
    int *hijk = NULL;
    int *hdiv = NULL;
    int *stride = NULL;
    int *dijk = NULL;
    int *ddiv = NULL;
    int check = 0;
    int rval = 0;

    started("i_to_ijk");

    // allocate pointers
    soln = (int*) malloc(dim*sizeof(int));
    hijk = (int*) malloc(dim*sizeof(int));
    hdiv = (int*) malloc(dim*sizeof(int));
    stride = (int*) malloc(dim*sizeof(int));
    cudaMalloc(&dijk, dim*sizeof(int));
    cudaMalloc(&ddiv, dim*sizeof(int));

    // setup
    for (i = 0; i < dim; ++i)
        hdiv[i] = 45;
    stride[0] = 1;
    for (i = 1; i < dim; ++i)
        stride[i] = stride[i-1]*hdiv[i-1];

    // copy constant data
    cudaMemcpy(ddiv, hdiv, dim*sizeof(int), cudaMemcpyHostToDevice);

    // try several values
    for (it = indices; *it != -1; ++it) {
        index = *it;
        // host evaluation
        for (i = dim; i != 0; --i) {
            hijk[i-1] = index / stride[i-1];
            index %= stride[i-1];
        }

        // device evaluation
        call_i_to_ijk<<<1, 1>>>(dijk, *it, ddiv);
        cudaMemcpy(soln, dijk, dim*sizeof(int), cudaMemcpyDeviceToHost);

        // check
        check = 0;
        for (i = 0; i < dim; ++i)
            if (hijk[i] != soln[i])
                check = 1;
        printf("  %d --> (%d", *it, hijk[0]);
        for (i = 1; i < dim; ++i)
            printf(", %d", hijk[i]);
        printf("): ");
        if (check) {
            printf("failed\n");
            rval = 1;
        } else {
            printf("passed\n");
        }
    }

    // cleanup
    free(soln);
    free(hijk);
    free(hdiv);
    free(stride);
    cudaFree(dijk);
    cudaFree(ddiv);

    return rval;
}

int test_distance_BpV() {
    int i = 0;
    int j = 0;
    const int natoms = 27;
    const int Nx = 200, Ny = 200, Nz = 200;
    // dynamic pointers
    dynamic_ptrs *dyn = NULL;
    // host
    float *hgrid = NULL;
    float *hpv = NULL;
    int *hndiv = NULL;
    float *hscaled_atom_pos = NULL;
    float *hatomic_radii = NULL;
    // device
    float *dgrid = NULL;
    float *dpv = NULL;
    int *dndiv = NULL;
    float *dscaled_atom_pos = NULL;
    float *datomic_radii = NULL;
    // timing
    clock_t start, stop;

    started("distance_BpV");

    // seed random number generator
    srandom(time(NULL));

    // allocate vectors
    construct(&dyn);
    hgrid = (float*) new_host(dyn, Nx*Ny*Nz*sizeof(float));
    hpv = (float*) new_host(dyn, DIM*DIM*sizeof(float));
    hndiv = (int*) new_host(dyn, DIM*sizeof(int));
    hscaled_atom_pos = (float*) new_host(dyn, DIM*natoms*sizeof(float));
    hatomic_radii = (float*) new_host(dyn, natoms*sizeof(float));
    dgrid = (float*) new_device(dyn, Nx*Ny*Nz*sizeof(float));
    dpv = (float*) new_device(dyn, DIM*DIM*sizeof(float));
    dndiv = (int*) new_device(dyn, DIM*sizeof(int));
    dscaled_atom_pos = (float*) new_device(dyn, DIM*natoms*sizeof(float));
    datomic_radii = (float*) new_device(dyn, natoms*sizeof(float));

    // setup
    memset(hpv, 0, DIM*DIM*sizeof(float));
    for (i = 0; i < DIM; ++i)
        hpv[i*DIM + i] = 10.0;
    hndiv[0] = Nx;
    hndiv[1] = Ny;
    hndiv[2] = Nz;
    for (i = 0; i < natoms; ++i)
        for (j = 0; j < DIM; ++j)
            hscaled_atom_pos[i*DIM + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < natoms; ++i)
        hatomic_radii[i] = 0.5*(float)(random())/RAND_MAX + 0.7;
    for (i = 0; i < Nx*Ny*Nz; ++i)
        hgrid[i] = 100000.0;

    // move data to the device and execute
    cudaMemcpy(dpv, hpv, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(1, dyn);
    cudaMemcpy(dndiv, hndiv, DIM*sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(2, dyn);
    cudaMemcpy(dscaled_atom_pos, hscaled_atom_pos, DIM*natoms*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(3, dyn);
    cudaMemcpy(datomic_radii, hatomic_radii, natoms*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(4, dyn);
    cudaMemcpy(dgrid, hgrid, Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(5, dyn);
    start = clock();
    distance_BpV<<<Nx*Ny*Nz, /* no./dim blocks */\
                   DIM*natoms, /* no./dim threads */\
                   (DIM*natoms+2*DIM)*sizeof(float)>>>(dgrid, Nx*Ny*Nz, \
                                                       dpv, dndiv, \
                                                       dscaled_atom_pos, \
                                                       datomic_radii, \
                                                       natoms);
    check_cuda(6, dyn);
    cudaDeviceSynchronize();
    stop = clock();
    {
        float ss = (float)(stop - start)/CLOCKS_PER_SEC;
        int mm = (int)(ss/60.);
        ss -= 60.0 * mm;
        printf("  distance_BpV completed in %02d:%09.6f\n", mm, ss);
    }
    cudaMemcpy(hgrid, dgrid, Nx*Ny*Nz*sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda(7, dyn);

    // print the data as a CHGCAR
    {
        FILE *ofs = fopen("CHGCAR", "w");
        fprintf(ofs, "%d randomly placed atoms in a %d x %d x %d grid\n", natoms, hndiv[0], hndiv[1], hndiv[2]);
        fprintf(ofs, " % 15.12f\n", 1.0);
        for (i = 0; i < DIM; ++i) {
            for (j = 0; j < DIM; ++j)
                fprintf(ofs, " % 15.12f", hpv[i*DIM + j]);
            fprintf(ofs, "\n");
        }
        fprintf(ofs, "C\n");
        fprintf(ofs, "%d\n", natoms);
        fprintf(ofs, "Selective Dynamics\n");
        for (i = 0; i < natoms; ++i) {
            for (j = 0; j < DIM; ++j) {
                fprintf(ofs, " % 12.8f", hscaled_atom_pos[i*DIM + j]);
            }
            fprintf(ofs, "\n");
        }
        fprintf(ofs, "\n");
        for (i = 0; i < DIM; ++i)
            fprintf(ofs, " %d", hndiv[i]);
        fprintf(ofs, "\n");
        for (i = 0; i < Nx*Ny*Nz; ++i)
            fprintf(ofs, "% 15.12f%s", hgrid[i], (((i+1)%5 == 0) ? "\n" : " "));
        if (i%5 != 5)
            fprintf(ofs, "\n");
        fclose(ofs);
    }

    printf("  Visually verify CHGCAR\n");

    // cleanup
    destroy(&dyn);

    return 0;
};

