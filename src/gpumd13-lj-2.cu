/*
* This code is courtesy of, and copyright 2015,
* Tomas Oppelstrup, Livermore National Lab. Please
* do not redistribute without his approval.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <unistd.h>

#include "real.h"

#define NBLOCKS  180
#define NTHREADS  64

#include "boxsortinc.cu"

double rawtime;

void vru(double r,double *v,double *g) {
  double rcut = 2.5;
  double alpha = -24.0*(pow(rcut,-7.0) - 2.0*pow(rcut,-13.0));
  double beta = -4.0*(pow(rcut,-12.0) - pow(rcut,-6.0));

  if(r < rcut && r > 0.1) {
    // Contribution to potential energy phi(r)
    *v = 4.0*(pow(r,-12.0) - pow(r,-6.0)) + alpha*(r - rcut) + beta;

    // Contribution to gradient 1/r * Dphi(r)/Dr
    *g = 24.0*(pow(r,-8.0) - 2.0*pow(r,-14.0)) + alpha/r;
  } else {
    *v = 0.0;
    *g = 0.0;
  }
}


void vru_grnpotential(double r,double *v,double *g) {
  /* %  Parameters: */
  const double a1 = 265.848;
  const double m  = 12;
  const double b1 = 1.5;
  const double c1 = 1.45;
  const double d  = 0.8;
  const double a2 = 2.5;
  const double b2 = 0.19;
  const double c2 = 1.89;

  double a1x,a2x;

  static int firsttime = 1;
  if(firsttime == 1) {
    printf("%% Potential parameters:\n"
	   "%%    a1 = %9.4f  a2 = %9.4f  m = %9.4f\n"
	   "%%    b1 = %9.4f  b2 = %9.4f  d = %9.4f\n"
	   "%%    c1 = %9.4f  c2 = %9.4f\n\n",
	   a1,a2,m,b1,b2,d,c1,c2);
    firsttime = 0;
  }

  /* %Formula: */
  if(r < 0.1) {
    *v = 0.0;
    *g = 0.0;
  } else if(r < c1) {
    a1x = a1*exp(b1/(r-c1));
    a2x = a2*exp(b2/(r-c2));
    *v = (1/pow(r,m)-d)*a1x + a2x;

    *g = (-m/pow(r,m+1) + (1/pow(r,m)-d)*(-b1/((r-c1)*(r-c1))))*a1x +
      a2x*(-b2/((r-c2)*(r-c2)));
    *g = *g/r;
  } else if(r < c2) {
    *v = a2*exp(b2/(r-c2));
    *g = *v * (-b2/((r-c2)*(r-c2)));
    *g = *g/r;
  } else {
    *v = 0;
    *g = 0;
  }
}


/* Transformation. From normal to skewed box:

     1    s/k  (s/k)^2
     0     1    s/k
     0     0     1

   Inverse transformation:

     1   -s/k    0
     0     1   -s/k
     0     0     1

*/

/* Figure out linear index of particle */
__host__ __device__
static int boxindex(real boxl,int k,volatile vector x, real g, real w) {
  //real g = 1.0/k, w = k/boxl;
  int a,b,c;
  a = (int) floor(x[0] * w);
  b = (int) floor((x[1] - g*x[0]) * w);
  c = (int) floor((x[2] - g*x[1]) * w);

  return a + k*(b + k*c);
}

__global__ void makeboxno(int n,int k,real boxl,vector4 xx[],int boxno[]) {
  const int pid = threadIdx.x + blockDim.x*blockIdx.x;
  const int  np = blockDim.x*gridDim.x;
  const int tid = threadIdx.x;
  const int  nt = blockDim.x;

  int k3 = k*k*k;
  real g = 1.0/k, w = k/boxl;
  volatile __shared__ struct {
    vector4 xx[NTHREADS];
  } shm;
  int i,bi;

  for(i = pid; i<n+tid; i+=np) {
    __syncthreads();
    shm.xx[0][tid+0*nt] = xx[i-tid][tid+0*nt];
    shm.xx[0][tid+1*nt] = xx[i-tid][tid+1*nt];
    shm.xx[0][tid+2*nt] = xx[i-tid][tid+2*nt];
    shm.xx[0][tid+3*nt] = xx[i-tid][tid+3*nt];
    __syncthreads();

    bi = boxindex(boxl,k,shm.xx[tid],g,w);
    bi = (k3 + (bi % k3)) % k3;
    if(i < n)
      boxno[i] = bi;
  }
}

/* Put particles in boxes */
static void boxem(int n,real boxl,int k,vector4 xx[],int first[],int perm[]) {
  int i,j,p,k3 = k*k*k;
  int *next;
  int bi;

  real g = 1.0/k, w = k/boxl;

  int *tags = (int *) alloca(sizeof(int) * n);

  memset(tags,0,sizeof(int) * n);
  next = (int *) alloca(n * sizeof(int));
  memset(next,0,sizeof(int) * n);
  memset(first,0,sizeof(int) * (k3+1));

  for(i = 0; i<n; i++) {
    bi = boxindex(boxl,k,xx[i],g,w);
    j = bi % k3;
    j = (k3+j)%k3;

    next[i] = first[j];
    first[j] = i+1;
  }

  i = 0;
  for(j = 0; j<k3; j++) {
    int ix = (i<n) ? i : i-n;
    p = first[j]-1;
    first[j] = ix; /*printf("First in box %2d is %2d. Chain is %2d",j,i,p);*/
    while(p >= 0) {
      tags[p] = tags[p] + 1;
      perm[i] = p; /*printf("location %3d has particle %3d.\n",i,p);*/
      i = i + 1;
      p = next[p]-1;
      /*printf(" %d",p);*/
    }
    /*printf("\n");*/
  }
  if(n != i) printf("* Serious counting error @%s:%d. i=%d n=%d k3=%d\n",
		    __FILE__,__LINE__,i,n,k3);

  for(i = 0; i<n; i++)
    if(tags[i] != 1) printf("Wrong tag: tags(%d) = %d\n",i,tags[i]);

  first[k3] = 0;
}

static void forcecalc_host(int n,real boxl,
			   int k,int first[],int boxno[],vector4 xx1[],
			   vector4 vv1[],vector4 xx2[],vector4 vv2[],real dt,
			   double *u_p,double *w_p,double *k_p,
			   int npot,double rcut,real upot[],real fpot[]) {
  double boxli = 1.0/boxl;

  int k3 = k*k*k;
  int i,j,i0,i1,j0,j1,iu,iv,b,ii;
  double xi,yi,zi,fxi,fyi,fzi,dx,dy,dz;
  double d2;
  double vx0,vy0,vz0,vx1,vy1,vz1,kx,ky,kz;
  double vr,u,rcut2 = rcut*rcut;
  double utot,wtot,ktot;

  utot = 0.0;
  wtot = 0.0;
  ktot = 0.0;

  for(b = 0; b<k3; b++) {
      i0 = first[b];
      i1 = first[b+1];

      for(i = i0; i!=i1; i++) {
	xi = xx1[i][0];
	yi = xx1[i][1];
	zi = xx1[i][2];
	ii = (int) xx1[i][3];

	fxi = 0.0;
	fyi = 0.0;
	fzi = 0.0;

	for(iv = -2; iv<=2; iv++)
	  for(iu = -2; iu<=2; iu++) {
	    j0 = (k3 + b + k*(iu + k*iv) - 2)%k3;

	    j1 = j0 + 5;
	    if(j1 >= k3) j1 = j1-k3;
	    j0 = first[j0];
	    j1 = first[j1];

	    if(j0 > n || j1 > n) {
	      printf("Crap in forcecalc_host :: n=%d j0=%d j1=%d\n",
		     n,j0,j1);
	      fflush(stdout);
	      exit(1);
	    }
	    if(j0 == n) j0 = 0;
	    if(j1 == n) j1 = 0;

	    for(j = j0; j!=j1; j=((j==n-1) ? 0 : j+1)) {
	      dx = xi - xx1[j][0];
	      dy = yi - xx1[j][1];
	      dz = zi - xx1[j][2];

	      dx = dx - boxl*rint(dx*boxli);
	      dy = dy - boxl*rint(dy*boxli);
	      dz = dz - boxl*rint(dz*boxli);
	      d2 = dx*dx + dy*dy + dz*dz;

	      if(d2 > 0.0 && d2 < rcut2) {
		//vru(sqrt(d2),&u,&vr);
		double fdx = d2/rcut2 * (npot-1);
		int idx = (int) floor(fdx);
		double frac = fdx-idx;
		//frac = floor(256.0*frac)/256.0;
		if(idx >= npot-1) {
		  u  = 0.0;
		  vr = 0.0;
		} else {
		  u  = (1.0-frac)*upot[idx] + frac*upot[idx+1];
		  vr = (1.0-frac)*fpot[idx] + frac*fpot[idx+1];
		}

		fxi = fxi - vr*dx;
		fyi = fyi - vr*dy;
		fzi = fzi - vr*dz;

		utot = utot + u;
		wtot = wtot - vr*d2;
	      }
	    }
	  }

	vx0 = vv1[i][0];
	vy0 = vv1[i][1];
	vz0 = vv1[i][2];
	vx1 = vx0 + fxi*dt;
	vy1 = vy0 + fyi*dt;
	vz1 = vz0 + fzi*dt;

	kx = vx0 + vx1;
	ky = vy0 + vy1;
	kz = vz0 + vz1;
	kx = kx*kx;
	ky = ky*ky;
	kz = kz*kz;

	ktot = ktot + (kx + ky + kz)*0.125;

	vv2[i][0] = vx1;
	vv2[i][1] = vy1;
	vv2[i][2] = vz1;

	xx2[i][0] = xi + dt*vx1;
	xx2[i][1] = yi + dt*vy1;
	xx2[i][2] = zi + dt*vz1;
	xx2[i][3] = ii;
      }
      }
  *u_p = utot*0.5;
  *w_p = wtot*0.5;
  *k_p = ktot;
}

__global__ static void adjustx4(int n,int k3,int first[],vector4 xx[],
				int startstop[]) {
  const int pid = threadIdx.x + blockDim.x*blockIdx.x;
  const int  np = blockDim.x*gridDim.x;
  int i,b,n4;
  real xi;

  __syncthreads();
  if(threadIdx.x == 0)
    startstop[blockIdx.x] = (startstop[blockIdx.x] == 0);

  for(b = pid; b<k3; b+=np) {
    i = first[b];
    first[b+k3  ] = i+  n;
    first[b+2*k3] = i+2*n;
  }
  n4 = 4*n;
  for(b = pid; b<n4; b+=np) {
    xi = xx[0][b];
    xx[n  ][b] = xi;
    xx[2*n][b] = xi;
  }

  __syncthreads();

  if(threadIdx.x == 0)
    startstop[gridDim.x+blockIdx.x] = (startstop[gridDim.x+blockIdx.x] == 0);
}

static texture<float2,1,cudaReadModeElementType> pottex;

__global__ static void
forcecalc_box(int n,real boxl,int k,int first[],int boxno[],vector4 xx1[],
	      vector4 vv1[],vector4 xx2[],vector4 vv2[],real dt,real ukout[],
	      int startstop[],real rcut,int npot) {
  volatile __shared__ int offsets[32];
  volatile __shared__ int j0share[NTHREADS],j1share[NTHREADS];
  volatile __shared__ real xxshare[NTHREADS][4];

  /*#define YYLEN (NTHREADS+160)*/
  #define YYLEN (NTHREADS+160)
  volatile __shared__ real yyshare[YYLEN][4];

  const int pid = threadIdx.x + blockDim.x*blockIdx.x;
  const int np  = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const int bid = blockIdx.x;
  const int nb  = gridDim.x;

  const int k3 = k*k*k;
  const real boxli = 1.0/boxl,rcut2 = rcut*rcut,potscale = (npot-1)/rcut2;
  //const real g = 1.0/k, w = k/boxl;

  real dx,dy,dz,d2,fxi,fyi,fzi;
  real utot,wtot,ktot,vx0,vx1;
  int i,j,j0,j1,iv,b;

  __syncthreads();
  if(tid == 0) startstop[bid] = (startstop[bid] == 0);

  for(i = tid; i<25; i+=nt)
    offsets[i] = (i/5-2)*k*k + (i%5-2)*k;

  utot = 0.0; wtot = 0.0; ktot = 0.0;
  for(i = pid; i<n+tid; i+=np) {
    // Load i-particles into shared memory, one particle per thread
    __syncthreads();
    xxshare[0][tid+0*nt] = xx1[i-tid][tid+0*nt];
    xxshare[0][tid+1*nt] = xx1[i-tid][tid+1*nt];
    xxshare[0][tid+2*nt] = xx1[i-tid][tid+2*nt];
    xxshare[0][tid+3*nt] = xx1[i-tid][tid+3*nt];
    __syncthreads();

    fxi = 0.0; fyi = 0.0; fzi = 0.0;


    // Loop over 25 neighboring columns
    b = n-1;
    if(i < n) b = i;
      b = boxno[b];//((boxindex(boxl,k,xxshare[tid],g,w)%k3)+k3)%k3;
    for(iv = 0; iv<25; iv++) {
      __syncthreads();
      j0share[tid] = first[k3+b+offsets[iv]-2];
      j1share[tid] = first[k3+b+offsets[iv]+2+1];
      __syncthreads();

      j0 = j0share[0]; j1 = j1share[nt-1];

      {
	int joff;
	for(joff = 0; joff<j1-j0; joff+=YYLEN) {
	  int jcount = j1-j0-joff;
	  if(jcount > YYLEN) jcount = YYLEN;

	  __syncthreads();
	  for(j = tid; j<4*jcount; j+=nt)
	    yyshare[0][j] = xx1[j0+joff][j];
	  __syncthreads();

	  {
	    int j0loc = j0share[tid] - j0share[0];
	    int j1loc = j1share[tid] - j0share[0];
	    if(j0loc < joff) j0loc = joff;
	    if(j1loc > joff+jcount) j1loc = joff+jcount;
	    for(j = j0loc; j<j1loc; j++) {
	      dx = xxshare[tid][0] - yyshare[j-joff][0];
	      dy = xxshare[tid][1] - yyshare[j-joff][1];
	      dz = xxshare[tid][2] - yyshare[j-joff][2];

	      dx = dx - boxl*rint(dx*boxli);
	      dy = dy - boxl*rint(dy*boxli);
	      dz = dz - boxl*rint(dz*boxli);
	      d2 = dx*dx + dy*dy + dz*dz;

	      if(d2 > 0.0 && d2 < rcut2) {
		float2 f = tex1D(pottex,0.5 + d2*potscale);

		fxi = fxi - f.y*dx;
		fyi = fyi - f.y*dy;
		fzi = fzi - f.y*dz;
		utot = utot + f.x;
		wtot = wtot - f.y*d2;
	      }
	    }
	  }
	}
      }
    }
    __syncthreads();
    for(j = 0; j<4; j++)
      yyshare[0][tid+j*nt] = vv1[i-tid][tid+j*nt];
    __syncthreads();

    if(i<n) {
      vx0 = yyshare[tid][0];
      vx1 = vx0 + fxi*dt;  vx0 = vx0 + vx1;
      ktot = ktot + vx0*vx0;
      yyshare[tid][0] = vx1;
      xxshare[tid][0] = xxshare[tid][0] + vx1*dt;

      vx0 = yyshare[tid][1];
      vx1 = vx0 + fyi*dt;  vx0 = vx0 + vx1;
      ktot = ktot + vx0*vx0;
      yyshare[tid][1] = vx1;
      xxshare[tid][1] = xxshare[tid][1] + vx1*dt;

      vx0 = yyshare[tid][2];
      vx1 = vx0 + fzi*dt;  vx0 = vx0 + vx1;
      ktot = ktot + vx0*vx0;
      yyshare[tid][2] = vx1;
      xxshare[tid][2] = xxshare[tid][2] + vx1*dt;
    }
    __syncthreads();
    for(j = tid; j<4*min(nt,n-(i-tid)); j+=nt) {
      xx2[i-tid][j] = xxshare[0][j];
      vv2[i-tid][j] = yyshare[0][j];
    }
  }

  __syncthreads();
  xxshare[0][tid+0*nt] = utot*0.5;
  xxshare[0][tid+1*nt] = wtot*0.5;
  xxshare[0][tid+2*nt] = ktot*0.125;
  __syncthreads();
  j = 1;
  while(j < nt) {
    i = (tid-j) | (j-1);
    if(tid & j) {
      xxshare[0][tid+0*nt] = xxshare[0][tid+0*nt] + xxshare[0][i+0*nt];
      xxshare[0][tid+1*nt] = xxshare[0][tid+1*nt] + xxshare[0][i+1*nt];
      xxshare[0][tid+2*nt] = xxshare[0][tid+2*nt] + xxshare[0][i+2*nt];
    }
    j = j<<1;
    __syncthreads();
  }

  for(i = tid; i<3; i+=nt)
    ukout[bid+i*nb] = xxshare[0][(i+1)*nt-1];

  __syncthreads();
  if(tid == 0) startstop[bid+nb] = (startstop[bid+nb] == 0);
  /*bad_exit:*/ __syncthreads();
  #undef YYLEN
}

double gettime(void) {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}


static int devChoice = -1;
int init_graphics_card(void) {
  /* Initialize graphics card */
  static int inited = 0;
  int devCount;

  if(inited == 0) {
    cudaGetDeviceCount(&devCount);
    if(devCount < 1) {
      printf("No devices...\n");
      exit(1);
    }
    if(devChoice < 0 || devChoice >= devCount) devChoice = devCount-1;
    printf("%% Number of devices is %d. Choosing device %d!\n", devCount,devChoice);
    cudaSetDevice(devChoice);
    inited = 1;
  }
  return 0;
}


int errorcheck(char s[],int nblocks,int startstop[]) {

  int i,err = 0;
  for(i = 0; i<2*nblocks; i++) if(startstop[i] != 1) err = err + 1;
  if(err) {
    printf("%s\n",s);
    printf("Error running kernel, errorcount = %d, nblocks = %d\n",
	   err,nblocks);
    printf("BLOCK: ");
    for(i = 0; i<nblocks; i++)
      printf("%4d",i);
    printf("\nSTART: ");
    for(i = 0; i<nblocks; i++)
      printf("%4d",startstop[i]);
    printf("\nSTOP :");
    for(i = 0; i<nblocks; i++)
      printf("%4d",startstop[nblocks+i]);
    printf("\n");
  }
  return err != 0;
}

int cardtimestep_box(int n,int k,vector4 xx[],vector4 vv[],real boxl,real dt,
		     double *utot,double *wtot,double *ktot,
		     int npot,real rcut,real upot[],real fpot[],
		     int coord_in,int coord_out) {
  static vector4 *xx1_dev,*xx2_dev,*vv1_dev,*vv2_dev;
  static int *boxno1_dev,*boxno2_dev,*first_dev,*startstop_dev;
  static real *uk_dev;
  static int ninit = 0, npotinit = 0;

  static cudaChannelFormatDesc channelDesc;
  static cudaArray *potarray;

  const int align = 32, nthreads = NTHREADS, nblocks = NBLOCKS;
  int k3 = k*k*k;

  int i;
  real *uk = (real *) alloca(sizeof(real) * 3*nblocks);
  int *startstop = (int *) alloca(sizeof(int) * 2*nblocks);

  if(ninit != n || npotinit != npot) {
    if(ninit > 0) {
      cudaFree(uk_dev);
      cudaFree(startstop_dev);
      cudaFree(first_dev);
      cudaFree(boxno2_dev);
      cudaFree(boxno1_dev);
      cudaFree(vv2_dev);
      cudaFree(vv1_dev);
      cudaFree(xx2_dev);
      cudaFree(xx1_dev);
    } else {
      init_graphics_card();
    }

    if(n > 0) {
      void *ptr;
      cudaMalloc(&ptr,(sizeof(vector4)*3*n + align-1)/align*align);
      xx1_dev = (vector4 *) ptr;
      cudaMalloc(&ptr,(sizeof(vector4)*3*n + align-1)/align*align);
      xx2_dev = (vector4 *) ptr;
      cudaMalloc(&ptr,(sizeof(vector4)*3*n + align-1)/align*align);
      vv1_dev = (vector4 *) ptr;
      cudaMalloc(&ptr,(sizeof(vector4)*3*n + align-1)/align*align);
      vv2_dev = (vector4 *) ptr;

      cudaMalloc(&ptr,(sizeof(int)*n + align-1)/align*align);
      boxno1_dev = (int *) ptr;
      cudaMalloc(&ptr,(sizeof(int)*n + align-1)/align*align);
      boxno2_dev = (int *) ptr;

      cudaMalloc(&ptr,(sizeof(int)*(k3+1)*3 + align-1)/align*align);
      first_dev = (int *) ptr;

      cudaMalloc(&ptr,(sizeof(real)*3*nblocks + align-1)/align*align);
      uk_dev = (real *) ptr;
      cudaMalloc(&ptr,(sizeof(int)*2*nblocks + align-1)/align*align);
      startstop_dev = (int *) ptr;

      channelDesc = cudaCreateChannelDesc<float2>();
      cudaMallocArray(&potarray,&channelDesc,npot,1);
      cudaMalloc(&ptr,sizeof(float2)*npot);

      {
	float2 *pcopy = (float2 *) alloca(sizeof(float2)*npot);
	for(i = 0; i<npot; i++) {
	  pcopy[i].x = upot[i];
	  pcopy[i].y = fpot[i];
	}
	cudaMemcpyToArray(potarray,0,0,pcopy,npot*sizeof(float2),
			   cudaMemcpyHostToDevice);
	pottex.addressMode[0] = cudaAddressModeClamp;
	pottex.filterMode = cudaFilterModeLinear;
	pottex.normalized = false;
	cudaBindTextureToArray(pottex,potarray,channelDesc);
      }
    }
    ninit = n; npotinit = npot;
  }

  if(n > 0) {
    double t0,t1;
    //printf("coord_in = %d , coord_out = %d\n",coord_in,coord_out);

    if(coord_in) {
      cudaMemcpy(xx1_dev,xx,sizeof(vector4) * n,cudaMemcpyHostToDevice);
      cudaMemcpy(vv1_dev,vv,sizeof(vector4) * n,cudaMemcpyHostToDevice);
    }

    for(i = 0; i<3*nblocks; i++) uk[i] = 0.0;
    cudaMemcpy(uk_dev,uk,sizeof(real) * 3*nblocks, cudaMemcpyHostToDevice);

    for(i = 0; i<2*nblocks; i++) startstop[i] = 0;
    cudaMemcpy(startstop_dev,startstop,sizeof(int)*2*nblocks,
	       cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    t0 = gettime();

    //printf("Computing box indices\n");
    makeboxno<<<nblocks,nthreads>>>(n,k,boxl,xx1_dev,boxno1_dev);

    /* Check box indices */ if(0) {
      int *boxno = (int *) malloc(sizeof(int) * n),nerr = 0;
      vector4 *xxtemp = (vector4 *) malloc(sizeof(vector4) * n);
      int *tags = (int *) malloc(sizeof(int) * n);
      cudaMemcpy(boxno,boxno1_dev,sizeof(int)*n,cudaMemcpyDeviceToHost);
      cudaMemcpy(xxtemp,xx1_dev,sizeof(vector4)*n,cudaMemcpyDeviceToHost);
      //printf("Checking box computation\n");
      for(i = 0; i<n; i++) {
	int bi = boxindex(boxl,k,xxtemp[i],1.0/k,k/boxl);
	bi = (k3 + (bi % k3)) % k3;
	if(boxno[i] != bi || bi<0 || bi>=k3) if(nerr++ < 10)
	  printf("boxno[%d] = %d, boxindex=%d\n",i,boxno[i],bi);
      }

      for(i = 0; i<n; i++) tags[i] = 0;
      for(i = 0; i<n; i++) tags[(int) xxtemp[i][3]]++;
      for(i = 0; i<n; i++)
	if(tags[i] != 1) if(nerr++ < 10) printf("input tag error: tag[%d] = %d\n",i,tags[i]);

      free(tags);
      free(xxtemp);
      free(boxno);
      if(nerr > 5) exit(1);
    }

    //printf("Sorting particles\n");
    rsort_card(n,k3+1,
	       boxno1_dev,xx1_dev,vv1_dev,
	       boxno2_dev,xx2_dev,vv2_dev,first_dev);
    /* Check sorting */ if(0) {
      int *boxno = (int *) malloc(sizeof(int) * n);
      int *first = (int *) malloc(sizeof(int) * (k3+1));
      vector4 *xxtemp = (vector4 *) malloc(sizeof(vector4) * n);
      int *tags = (int *) malloc(sizeof(int) * n);
      int nerr = 0;
      cudaMemcpy(boxno,boxno2_dev,sizeof(int)*n,cudaMemcpyDeviceToHost);
      cudaMemcpy(first,first_dev,sizeof(int)*(k3+1),cudaMemcpyDeviceToHost);
      cudaMemcpy(xxtemp,xx2_dev,sizeof(vector4)*n,cudaMemcpyDeviceToHost);
      //printf("Checking sorting\n");
      for(i = 1; i<n; i++) {
	if(boxno[i]<boxno[i-1]) if(nerr++ < 10)
	  printf("Sorting error: boxno[%d] = %d, boxno[%d]=%d\n",
		 i,boxno[i],i-1,boxno[i-1]);
      }

      for(i = 0; i<n; i++) tags[i] = 0;
      for(i = 0; i<n; i++) tags[(int) xxtemp[i][3]]++;
      for(i = 0; i<n; i++)
	if(tags[i] != 1) if(nerr++ < 10) printf("tag error: tag[%d] = %d\n",i,tags[i]);

      //printf("n=%d k3=%d first[0]=%d  first[k3-1]=%d first[k3]=%d\n",
      //     n,k3,first[0],first[k3-1],first[k3]);
      for(i = 0; i<k3; i++) {
	int j;
	for(j = first[i]; j<first[i+1]; j++)
	  if(boxno[j] != i) if(nerr++ < 10)
	    printf("first/box error: boxno[%d]=%d first[%d]=%d first[%d]=%d\n",
		   j,boxno[j],i,first[i],i+1,first[i+1]);
	if(first[i+1] - first[i] > 15) {
	  printf("Very full box %d: %d\n",i,first[i+1]-first[i]);
	  for(j = first[i]; j<first[i+1]; j++) {
	    printf("particle %5d in box %4d :: %10.3f %10.3f  %10.3f  %10.2f\n",
		   j,i,xxtemp[j][0],xxtemp[j][1],xxtemp[j][2],xxtemp[j][3]);
	  }
	  exit(1);
	}
      }
      free(tags);
      free(xxtemp);
      free(first);
      free(boxno);
      if(nerr > 0) exit(1);
    }

    //printf("Running adjust4x\n");
    adjustx4<<<nblocks,nthreads/*,sizeof(vector4)*(nthreads+5)*5*/>>>
      (n,k3,first_dev,xx2_dev,startstop_dev);
    cudaThreadSynchronize();

    for(i = 0; i<2*nblocks; i++) startstop[i] = 0;
    cudaMemcpy(startstop,startstop_dev,sizeof(int)*2*nblocks,
	       cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    if(errorcheck("KERNEL ADJUSTX4",nblocks,startstop)) { exit(1); }

    t1 = gettime();
    //rawtime = t1-t0;
    //cudaMemcpy(first_dev,first,sizeof(int) * (k3+1),cudaMemcpyHostToDevice);

    *utot = 0.0; *wtot = 0.0; *ktot = 0.0;

    for(i = 0; i<2*nblocks; i++) startstop[i] = 0;
    cudaMemcpy(startstop_dev,startstop,sizeof(int)*2*nblocks,
	       cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    //printf("Running force calculation\n");
    t0 = gettime();
    forcecalc_box<<<nblocks,nthreads>>>(n,boxl,k,first_dev,boxno2_dev,
					xx2_dev,vv2_dev,xx1_dev,vv1_dev,
					dt,uk_dev,startstop_dev,rcut,npot);
    cudaThreadSynchronize();
    t1 = gettime();
    rawtime = t1-t0;
    //printf("Force caculation done.\n");
    //printf("%120s Rawtime: %.3f ms\n","",rawtime*1e3);

    cudaMemcpy(startstop,startstop_dev,sizeof(int)*2*nblocks,
	       cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    if(errorcheck("KERNEL FORCECALC_BOX",nblocks,startstop)) { exit(1); }

    cudaMemcpy(uk,uk_dev,sizeof(real  ) * 3*nblocks, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    if(coord_out) {
      /*int nerr = 0;*/
      int *tags = (int *) malloc(sizeof(int) * n);
      cudaMemcpy(xx,xx1_dev,sizeof(vector4) * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(vv,vv1_dev,sizeof(vector4) * n, cudaMemcpyDeviceToHost);

      /*
      for(i = 0; i<n; i++) tags[i] = 0;
      for(i = 0; i<n; i++) tags[(int) xx[i][3]]++;
      for(i = 0; i<n; i++)
	if(tags[i] != 1) if(nerr++ < 5) printf("force tag error (xx): tag[%d] = %d\n",i,tags[i]);

      nerr = 0;
      for(i = 0; i<n; i++) tags[i] = 0;
      for(i = 0; i<n; i++) tags[(int) vv[i][3]]++;
      for(i = 0; i<n; i++)
	if(tags[i] != 1) if(nerr++ < 5) printf("force tag error (vv): tag[%d] = %d\n",i,tags[i]);
      */
      free(tags);
    }
    for(i = 0; i<nblocks; i++) {
      //if(uk[i] > *utot) *utot = uk[i];
      *utot = *utot + uk[i+0*nblocks];
      *wtot = *wtot + uk[i+1*nblocks];
      *ktot = *ktot + uk[i+2*nblocks];
    }

  }
  return 0;
}

void bswap(int n, int sz, void *v) {
  char *p = (char *) v;
  char t;
  int i,k;

  for(i = 0; i<n; i++)
    for(k = 0; k<sz/2; k++) {
      t = p[i*sz + k];
      p[i*sz + k] = p[i*sz + sz-k-1];
      p[i*sz + sz-k-1] = t;
    }
}

void storecfg(char *fname,int n,vector *xx,int byteorder) {
  double *xout = (double *) malloc(sizeof(double) * 3*n);
  int i,j,len;
  FILE *f;

  f = fopen(fname,"w");
  if(f == NULL) {
    printf("Can not open file %s for writing.\n",fname);
    free(xout);
    return;
  }

  len = 3*n*sizeof(double);
  for(i = 0; i<n; i++)
    for(j = 0; j<3; j++)
      xout[n*j+i] = xx[i][j];

  if(byteorder) {
    bswap(1,sizeof(int),&len);
    bswap(3*n,sizeof(double),xout);
  }

  fwrite(&len,sizeof(int),1,f);
  fwrite(xout,3*sizeof(double),n,f);
  fwrite(&len,sizeof(int),1,f);
  fclose(f);

  free(xout);
}

int loadcfg(char *fname,vector **xx,int *byteorder) {
  FILE *f;
  int n,do_swap,len;
  double *xin;
  int i,j;

  f = fopen(fname,"r");
  if(f == NULL) {
    printf("Can not open file %s for reading.\n",fname);
    return -1;
  }

  fseek(f,0,SEEK_END);
  len = ftell(f);
  fseek(f,0,SEEK_SET);

  fread(&n,sizeof(int),1,f);
  if(len != (int) (n+2*sizeof(int))) {
    bswap(1,sizeof(int),&n);
    if(len != (int) (n+2*sizeof(int))) {
      printf("Crap, unable to understand md.cfg\n");
      fclose(f);
      return -1;
    }
    do_swap = 1;
  } else do_swap = 0;
  n = n / (3*sizeof(double));

  ///printf("do_swap = %d     n = %d\n",do_swap,n);

  *xx = (vector  *) malloc(sizeof(vector ) * n);
  xin = (double *) malloc(sizeof(double) * 3*n);

  fread(xin,sizeof(double)*3,n,f);
  if(do_swap) bswap(3*n,sizeof(double),xin);
  for(i = 0; i<n; i++)
    for(j = 0; j<3; j++)
      (*xx)[i][j] = xin[n*j+i];
  free(xin);

  fread(&len,sizeof(int),1,f);
  fclose(f);

  if(do_swap) bswap(1,sizeof(int),&len);
  if(len != (int) (sizeof(double)*3*n)) {
    printf("Crap, unable to understand file %s (stage two) %d %d\n",
	   fname,len,(int) (sizeof(double)*3*n));
    free(xx);
    return -1;
  }

  *byteorder = do_swap;
  return n;
}

int main(int argc, char *argv[]) {
  int niter,nrescale,noutput,nrestart,ncompare,nmomentum,cfgsave;
  int iter0,iter;
  int i,j,k,k3,n,nin;
  real boxl,dt;
  vector  *xx,*vv,*xx0;
  vector4 *xx4,*vv4,*xx4save,*vv4save;
  //int *first,*perm;
  double utot,wtot,ktot,p,tinst,etotlast = 0.0;;
  double rho,rhoguess;
  double tfixed;
  double Uavg,Tavg,Pavg,Tscaleavg,msd = 0.0;
  FILE *logfile;
  char line[100];
  int byteorder = 0,echange;

  real rcut = 2.51;
  int npot = 1000;
  real *upot = (real *) alloca(sizeof(real)*npot);
  real *fpot = (real *) alloca(sizeof(real)*npot);

  int coord_in,coord_out;

  if(argc >= 3 && strcmp(argv[1],"-device") == 0) {
    devChoice = atoi(argv[2]);
    printf("%% Command line option set tentative device number %d\n",devChoice);
  }

  /* Compute potantial table */
  for(i = 0; i<npot; i++) {
    double v,g;
    double r2 = i*rcut*rcut/(npot-1);
    vru(sqrt(r2),&v,&g);
    upot[i] = v;
    fpot[i] = g;
  }

  /* Load initial configuration */
  n = loadcfg("md.cfg",&xx,&byteorder);
  n = loadcfg("md.vel",&vv,&byteorder);
  {
    FILE *fp = fopen("md0.cfg","r");
    if(fp == NULL) {
      xx0 = (vector *) malloc(sizeof(vector) * n);
      memcpy(xx0,xx,sizeof(vector)*n);
      storecfg("md0.cfg",n,xx0,byteorder);
    } else {
      fclose(fp);
      n = loadcfg("md0.cfg",&xx0,&byteorder);
    }
  }

  {
    FILE *fp = fopen("md.inp","r");

    if(fp == NULL) {
      printf("Cannot open input file md.inp\n");
      exit(1);
    }
    fgets(line,sizeof(line),fp); sscanf(line+29,"%d",&nin);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%lf",&rho);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%lf",&tfixed);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%d",&nrescale);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%f",&dt);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%d",&niter);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%d",&noutput);
    fgets(line,sizeof(line),fp); sscanf(line+29,"%d",&nrestart);
    fgets(line,sizeof(line),fp); // potential cut off
    fgets(line,sizeof(line),fp); // cubic flag
    fgets(line,sizeof(line),fp); // noncubic data
    fgets(line,sizeof(line),fp); // noncubic data
    fgets(line,sizeof(line),fp); // noncubic data
    fgets(line,sizeof(line),fp); // mpi data
    fgets(line,sizeof(line),fp); // mpi data
    fgets(line,sizeof(line),fp); // pot file
    fgets(line,sizeof(line),fp);  sscanf(line+29,"%d",&cfgsave);
    boxl  = pow(n/rho,1.0/3.0);
  }
  {
    FILE *fp = fopen("md.sts","r");
    iter0 = 1; Uavg = 0.0; Tavg = 0.0; Pavg = 0.0; Tscaleavg = 0.0;
    if(fp == NULL) {
      fp = fopen("md.sts","w");
      fprintf(fp,"%12d %20.10e %20.10e %20.10e %20.10e\n",
	      iter0,Uavg,Tavg,Pavg,Tscaleavg);
      fclose(fp);
    } else {
      fscanf(fp,"%d%lf%lf%lf%lf",&iter0,&Uavg,&Tavg,&Pavg,&Tscaleavg);
      Uavg = Uavg * ((iter0-1) % noutput);
      Tavg = Tavg * ((iter0-1) % noutput);
      Pavg = Pavg * ((iter0-1) % noutput);
    }
  }
  logfile = fopen("md.log","a");

  /* Compute number of boxes to divide system into */
  k = (int) floor(2*boxl/rcut);
  while(k>0 && k+boxl/(4*k*k*rcut) > 2*boxl/rcut) k=k-1;
  if(k <= 0) {
    printf("Error in k, k=%d boxl=%f rcut=%f\n",k,boxl,rcut);
    exit(1);
  }
  k3 = k*k*k;

  /* Compute an estimate of the particle density */
  {
    double xmax = -1e20;
    for(i = 0; i<n; i++)
      for(j = 0; j<3; j++)
	if(xx[i][j] > xmax) xmax = xx[i][j];
    rhoguess = n/(xmax*xmax*xmax);
  }

  if(fabs(rhoguess-rho) > 1e-3)
    printf("WARNING, rho and rhoguess differ with more than 1e-3.\n");

  if(n != nin)
    printf("WARNING, N in cfgfile and md.inp differ.\n");

  ncompare = 1000000000;   /* How often to compare cpu/card computations */
  nmomentum = 100;      /* How often to rescale momentu, (often due to single precision)*/

  printf("%% MD CONFIGURATION\n"
	 "%% n        = %7d\n"
	 "%% k        = %7d\n"
	 "%% k3       = %7d\n"
	 "%% rho      = %11.4f\n"
	 "%% rhoguess = %11.4f\n"
	 "%% boxl     = %15.8f\n"
	 "%% dt       = %15.8f\n"
	 "%% niter    = %9d\n"
	 "%% cardcmp  = %9d\n"
	 "%% momentum = %9d\n",
	 n,k,k3,rho,rhoguess,boxl,dt,niter,ncompare,nmomentum);

  /* Allocate memory for internal data structure */
  xx4save = (vector4 *) malloc(sizeof(vector4) * n);
  vv4save = (vector4 *) malloc(sizeof(vector4) * n);
  xx4     = (vector4 *) malloc(sizeof(vector4) * n);
  vv4     = (vector4 *) malloc(sizeof(vector4) * n);

  for(i = 0; i<n; i++) {
    for(j = 0; j<3; j++) {
      xx4[i][j] = xx[i][j];
      vv4[i][j] = vv[i][j];
    }
    xx4[i][3] = i;
    vv4[i][3] = i;
  }

  echange = 1;
  coord_out = 1;
  for(iter = iter0; iter<niter+iter0; iter++) {
    double t0,t1/*,boxtime*/;

    //t0 = gettime();
    /* Save configuration before timestep so that
       a step can be performed on the cpu, and so
       that it can be dumped to file in case of
       error */
    if(iter % ncompare == 0) {
      memcpy(xx4save,xx4,n*sizeof(vector4));
      memcpy(vv4save,vv4,n*sizeof(vector4));
    }

    if(coord_out) coord_in = 1; else coord_in = 0;

    coord_out = 0;
    if(iter % noutput == 0) coord_out = 1;
    if(iter % ncompare == ncompare-1 || iter % ncompare == 0) coord_out = 1;
    if(iter % nmomentum == 0) coord_out = 1;
    if(iter % nrestart == 0 || iter==iter0+niter-1) coord_out = 1;

    t0 = gettime();
    cardtimestep_box(n,k,xx4,vv4,boxl,dt,
		     &utot,&wtot,&ktot,
		     npot,rcut,upot,fpot,
		     coord_in,coord_out);
    t1 = gettime();

    if(iter % noutput == 0 ||  iter % ncompare == 0) {
      msd = 0.0;
      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  msd = msd + (xx0[(int) xx4[i][3]][j] - xx4[i][j])*(xx0[(int) xx4[i][3]][j] - xx4[i][j]);
    }

    utot = utot/n;
    wtot = wtot/(3*n);
    ktot = ktot/n;
    tinst = 2.0/3.0 * ktot;
    p = rho*(tinst + wtot);
    msd = msd/n;
    //t1 = gettime();

    /* If total energy changes by more than 1% in one iteration,
       that indicates a srious error. This codes dumps the state
       that produced the error. */
    if(0) if(echange == 0 &&
       fabs(etotlast-utot-ktot)>0.01*fabs(etotlast) &&
       fabs(etotlast-utot-ktot)>0.01) {
      char s[80];
      FILE *f;

      printf("%% card: %20.10e %20.10e %20.10e %20.10e %10.3f\n",
	     utot,ktot,utot+ktot,p,(t1-t0)*1e3);

      printf("%% Serious energy error. "
	     "Dumping configuration and exiting...\n");
      sprintf(s,"totaldump.%d",iter);
      f = fopen(s,"w");

      /* Simulation parameters */
      fwrite(&n,sizeof(n),1,f);
      fwrite(&k,sizeof(k),1,f);
      fwrite(&k3,sizeof(k3),1,f);
      fwrite(&boxl,sizeof(boxl),1,f);
      fwrite(&rcut,sizeof(rcut),1,f);
      fwrite(&dt,sizeof(dt),1,f);

      /* Input to time-step */
      fwrite(xx4save,sizeof(vector4),n,f);
      fwrite(vv4save,sizeof(vector4),n,f);
      fwrite(xx4,sizeof(vector4),n,f);
      fwrite(vv4,sizeof(vector4),n,f);

      /* Output from time-step */
      fwrite(xx,sizeof(vector),n,f);
      fwrite(vv,sizeof(vector),n,f);
      fclose(f);

      break;
    } else etotlast = utot + ktot;
    echange = 0;

    /* Output statistics */
    Uavg = Uavg + utot;
    Tavg = Tavg + tinst;
    Pavg = Pavg + p;
    Tscaleavg = Tscaleavg + tinst;
    if(iter % noutput == 0) {
      Uavg = Uavg / noutput;
      Tavg = Tavg / noutput;
      Pavg = Pavg / noutput;
      printf("%12d %20.10e %20.10e %20.10e %20.10e %20.10e\n",
	     iter,Uavg+Tavg*1.5,Uavg,Tavg,Pavg,msd);
      fprintf(logfile,
	      "%12d %20.10e %20.10e %20.10e %20.10e %20.10e\n",
	     iter,Uavg+Tavg*1.5,Uavg,Tavg,Pavg,msd);
      Uavg = 0.0; Tavg = 0.0; Pavg = 0.0;
    }

    etotlast = utot + ktot;

    if(iter % ncompare == 0) {
      /* Run same timestep on cpu, and pring statistics for both card
	 and cpu step, for accuracy comparisons. */
      printf("%% card: %12d %20.10e %20.10e %20.10e %20.10e %20.10e %10.3f\n",
	     iter,utot+ktot,utot,tinst,p,msd,(t1-t0)*1e3);
      fprintf(logfile,
	      "%% card: %12d %20.10e %20.10e %20.10e %20.10e %20.10e %10.3f\n",
	     iter,utot+ktot,utot,tinst,p,msd,(t1-t0)*1e3);

      t0 = gettime();

      {
	int *first = (int *) malloc(sizeof(int) * (k3+1));
	int *perm  = (int *) malloc(sizeof(int) * n);
	int *boxno = (int *) malloc(sizeof(int) * n);
	vector4 *xx4temp = (vector4 *) malloc(sizeof(vector4) * n);
	vector4 *vv4temp = (vector4 *) malloc(sizeof(vector4) * n);
	int jsave;

	//printf("%% -- CPU check. Running boxem...\n"); fflush(stdout);
	boxem(n,boxl,k,xx4save,first,perm);
	//printf("%% -- boxem complete\n"); fflush(stdout);
	jsave = k3;
	while(first[jsave] == 0) {
	  first[jsave] = n; jsave = jsave-1;
	}


	//printf("%% -- Copying to xx4temp\n"); fflush(stdout);
	for(i = 0; i<n; i++) {
	  for(j = 0; j<3; j++) {
	    xx4temp[i][j] = xx4save[perm[i]][j];
	    vv4temp[i][j] = vv4save[perm[i]][j];
	  }
	  xx4temp[i][3] = xx4save[perm[i]][3];
	  vv4temp[i][3] = xx4save[perm[i]][3];
	}
	//printf("%% -- Assigning to boxno\n"); fflush(stdout);
	for(i = 0; i<k3; i++)
	  for(j = first[i]; j<first[i+1]; j++)
	    boxno[j] = i;

	//printf("%% -- Calling forcecalc_host...\n"); fflush(stdout);
	forcecalc_host(n,boxl,k,first,boxno,xx4temp,
		       vv4temp,xx4save,vv4save,dt,
		       &utot,&wtot,&ktot,npot,rcut,upot,fpot);

	//printf("%% -- forcecalc_host complete\n"); fflush(stdout);
	free(vv4temp);
	free(xx4temp);
	free(boxno);
	free(perm);
	free(first);
      }

      //printf("%% -- Copmuting msd\n"); fflush(stdout);
      msd = 0.0;
      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  msd = msd + (xx0[(int) xx4save[i][3]][j] - xx4save[i][j])*
	    (xx0[(int) xx4save[i][3]][j] - xx4save[i][j]);
      //printf("%% -- msd calculation complete\n"); fflush(stdout);
      utot = utot/n;
      wtot = wtot/(3*n);
      ktot = ktot/n;
      tinst = 2.0/3.0 * ktot;
      p = rho*(tinst + wtot);
      msd = msd/n;
      t1 = gettime();

      printf("%%  cpu: %12d %20.10e %20.10e %20.10e %20.10e %20.10e %10.3f\n",
	     iter,utot+ktot,utot,tinst,p,msd,(t1-t0)*1e3);
      fprintf(logfile,
	      "%%  cpu: %12d %20.10e %20.10e %20.10e %20.10e %20.10e %10.3f\n",
	     iter,utot+ktot,utot,tinst,p,msd,(t1-t0)*1e3);
      fflush(stdout); fflush(logfile);
    }
    //printf("Quitting here... %s:%d\n",__FILE__,__LINE__);
    //exit(1);

    if(iter % nmomentum == 0) {
      double mom[3] = {0.0, 0.0, 0.0};
      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  mom[j] = mom[j] + vv4[i][j];
      /*printf("%% Momentum is (%20.10e , %20.10e , %20.10e)\n",
	mom[0],mom[1],mom[2]);*/
      for(j = 0; j<3; j++) mom[j] = mom[j] / n;
      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  vv4[i][j] = vv4[i][j] - mom[j];

      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++) {
	  double t = boxl*floor(xx4[i][j]/boxl);
	  xx4[i][j] = xx4[i][j] - t;
	  xx0[(int) xx4[i][3]][j] = xx0[(int) xx4[i][3]][j] - t;
	  xx[(int) xx4[i][3]][j] = xx4[i][j];
	  vv[(int) vv4[i][3]][j] = vv4[i][j];
	}


      /*
      for(j = 0; j<3; j++) mom[j] = 0.0;
      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  mom[j] = mom[j] + vv[i][j];
      */
      /*printf("%% Corrected   (%20.10e , %20.10e , %20.10e)\n",
	mom[0],mom[1],mom[2]);*/

      echange = 1;
    }



    if(nrescale > 0 && iter % nrescale == 0) {
      double alpha;

      Tscaleavg = Tscaleavg / nrescale;

      /* alpha = (2*tfixed - Tscaleavg)/Tscaleavg; */
      alpha = 1.0 + 1.8*(tfixed - Tscaleavg)/tinst;
      if(alpha < 1e-6) alpha = 1e-6;
      alpha = sqrt(alpha);

      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++)
	  vv4[i][j] = vv4[i][j]*alpha;

      Tscaleavg = 0.0;
      echange = 1;
    }

    if(iter % nrestart == 0 || iter==iter0+niter-1) {
      char fname[80];
      FILE *fp;

      for(i = 0; i<n; i++)
	for(j = 0; j<3; j++) {
	  double t = boxl*floor(xx4[i][j]/boxl);
	  xx4[i][j] = xx4[i][j] - t;
	  xx0[(int) xx4[i][3]][j] = xx0[(int) xx4[i][3]][j] - t;
	  xx[(int) xx4[i][3]][j] = xx4[i][j];
	  vv[(int) vv4[i][3]][j] = vv4[i][j];
	}

      fclose(logfile);

      if(cfgsave == 1){
	sprintf(fname,"md%09d.cfg",iter);
	storecfg(fname,n,xx,byteorder);
      }
      if(cfgsave == 2){
	sprintf(fname,"md%09d.cfg",iter);
	storecfg(fname,n,xx,byteorder);
	sprintf(fname,"md0_%09d.cfg",iter);
	storecfg(fname,n,xx0,byteorder);
	sprintf(fname,"md%09d.vel",iter);
	storecfg(fname,n,vv,byteorder);
      }
      storecfg("md.cfg",n,xx,byteorder);
      storecfg("md.vel",n,vv,byteorder);
      storecfg("md0.cfg",n,xx0,byteorder);
      fp = fopen("md.sts","w");
      fprintf(fp,"%12d %20.10e %20.10e %20.10e %20.10e\n",
	      iter+1,Uavg/iter,Tavg/iter,Pavg/iter,Tscaleavg);
      fclose(fp);
      logfile = fopen("md.log","a");
    }
  }

  /* Release memory allocated on graphics card */
  cardtimestep_box(-1,-1,NULL,NULL,
		   0.0,0.0,NULL,NULL,NULL,0,0.0,NULL,NULL,0,0);

  free(xx);
  free(vv);
  free(xx0);
  free(xx4);
  free(vv4);
  free(xx4save);
  free(vv4save);

  return 0;
}
