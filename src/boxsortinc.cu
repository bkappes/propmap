/*
* This code is courtesy of, and copyright 2015,
* Tomas Oppelstrup, Livermore National Lab. Please
* do not redistribute without his approval.
*/
#define NTHREADS_RADIX 128
#define NBLOCKS_RADIX   56

__global__ static void
boxsum_stage1(int nc,int count[]) {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = gridDim.x;
  const int pid = bid*nt + tid, np = nb*nt;
  int i;
  for(i = pid; i<nc; i+=np)
    count[i] = 0;
}

__global__ static void
boxsum_stage2(int n,int v[],int listid[],int count[]) {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = gridDim.x;
  const int pid = bid*nt + tid, np = nb*nt;
  int i,vi,x;
  volatile __shared__ struct { int vi[NTHREADS_RADIX],x[NTHREADS_RADIX]; } shm;

  if(0) {
    for(i = pid; i<n; i+=np) {
      // Compute index within box.
      vi = v[i];
      x = atomicAdd(&count[vi],1);
      listid[i] = x;
    }
  } else {
    for(i = pid; i<n+tid; i+=np) {
      // Compute index within box.
      if(n-(i-tid) < nt) {
	if(i < n) {
	  vi = v[i];
	  x = atomicAdd(&count[vi],1);
	  listid[i] = x;
	}
      } else {
	shm.vi[tid] = v[i];
	__syncthreads();
	// Requirement is that gcd(11,nt) = 1
	shm.x[(19*tid)%nt] = atomicAdd(&count[shm.vi[(19*tid)%nt]],1);
	__syncthreads();
	listid[i] = shm.x[tid];
      }
    }

    /*
    int i0,i1;
    {
      int q = n/np;
      int r = n%np;
      if(pid >= r) { i0 = q*pid + r; i1 = i0 + q; }
      else { i0 = q*pid + pid; i1 = i0 + q + 1; }
    }
    for(i = i0; i<i1; i++) {
      vi = v[i];
      x = atomicAdd(&count[vi],1);
      listid[i] = x;
    }
    */
  }
}

__global__ static void
boxsum_stage3(int nboxes,int count[],int psum[]) {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = gridDim.x;
  const int pid = bid*nt + tid, np = nb*nt;
  int i,x;
  volatile __shared__ struct { int x[NTHREADS_RADIX]; } shm;
  for(i = pid; i<nboxes+tid; i+=np) {
    __syncthreads();
    x = 0;
    if(i < nboxes) x = count[i];
    shm.x[tid] = x;
    __syncthreads();
    if(tid < 64) shm.x[tid] += shm.x[tid+64];
    __syncthreads();
    if(tid < 32) {
      shm.x[tid] += shm.x[tid+32];
      shm.x[tid] += shm.x[tid+16];
      shm.x[tid] += shm.x[tid+ 8];
      shm.x[tid] += shm.x[tid+ 4];
      shm.x[tid] += shm.x[tid+ 2];
      shm.x[tid] += shm.x[tid+ 1];
    }
    if(tid == 0) psum[i/nt] = shm.x[0];
  }
}

__global__ static void
boxsum_stage4(int n,int psum[]) {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x;
  int i,j,x,s;
  volatile __shared__ int xshare[NTHREADS_RADIX];

  s = 0;
  if(bid == 0)
    for(i = tid; i<n+tid; i+=nt) {
      __syncthreads();
      x = 0;
      if(i < n) x = psum[i];
      xshare[tid] = x;
      __syncthreads();
      // Make cumulative summation of columns in type!
      j = 1;
      while(j < nt) {
	if(tid >= j) x += xshare[tid-j];
	__syncthreads();
	xshare[tid] = x;
	j = j*2;
	__syncthreads();
      }
      if(i < n) psum[i] = xshare[tid] + s;
      s = s + xshare[nt-1];
    }
}

__global__ static void
boxsum_stage5(int nboxes,int count[],int psum[])  {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = NBLOCKS_RADIX;
  const int pid = bid*nt + tid, np = nb*nt;

  int i,x,x1;
  volatile __shared__ struct {
    int psum,x[2][NTHREADS_RADIX],y[2][NTHREADS_RADIX];
  } shm;

  shm.x[0][tid] = 0;
  shm.y[0][tid] = 0;
  for(i = pid; i<nboxes+tid; i+=np) {
    __syncthreads();
    if(tid == 0) shm.psum = psum[i/nt];
    x = 0;
    if(i < nboxes) x = count[i];
    x1 = x;
    shm.x[1][tid] = x; __syncthreads();
    x += shm.x[1][tid- 1]; shm.y[1][tid] = x; __syncthreads();
    x += shm.y[1][tid- 2]; shm.x[1][tid] = x; __syncthreads();
    x += shm.x[1][tid- 4]; shm.y[1][tid] = x; __syncthreads();
    x += shm.y[1][tid- 8]; shm.x[1][tid] = x; __syncthreads();
    x += shm.x[1][tid-16]; shm.y[1][tid] = x; __syncthreads();
    x += shm.y[1][tid-32]; shm.x[1][tid] = x; __syncthreads();
    x += shm.x[1][tid-64]; shm.y[1][tid] = x; __syncthreads();
    x += shm.psum - shm.y[1][nt-1];
    //if(i == nboxes-1) { x=0; x1=0; }
    if(i < nboxes) count[i] = x-x1;
  }
}

__global__ static void
boxsum_stage6(int n,int v[],int listid[],int count[])  {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = NBLOCKS_RADIX;
  const int pid = bid*nt + tid, np = nb*nt;
  int i,lid,bno,idx;
  for(i = pid; i<n; i+=np) {
    lid = listid[i];
    bno = v[i];
    idx = 0;
    if(bno > 0) idx = count[bno];
    listid[i] = lid + idx;
  }
}

__global__ static void
boxsum_stage7(int n,int listid[],int bin[],float xx[][4], float vv[][4],
	      int bout[],float xxout[][4], float vvout[][4]) {
  const int tid = threadIdx.x, nt = NTHREADS_RADIX;
  const int bid = blockIdx.x,  nb = NBLOCKS_RADIX;
  const int pid = bid*nt + tid, np = nb*nt;
  int i,j,xend;
  float x,v;
  volatile __shared__ struct { int idx[NTHREADS_RADIX]; } shm;
  for(i = pid; i<n+tid; i+=np) {
    j = 0;
    if(i < n) j = listid[i];
    xend = min(nt,n-(i-tid));
    __syncthreads();
    shm.idx[tid] = j;
    __syncthreads();

    if(i < n) bout[j] = bin[i];
    for(j = tid; j<4*xend; j+=nt) {
      x = xx[i-tid][j];
      v = vv[i-tid][j];
      xxout[shm.idx[j/4]][j%4] = x;
      vvout[shm.idx[j/4]][j%4] = v;
    }
  }
}

void rsort_card(int n,int nc,
		int *xin_g,float (*data1in_g)[4],float (*data2in_g)[4],
		int *xout_g,float (*data1out_g)[4],float (*data2out_g)[4],
		int *count_g) {
  static int n_init = 0, nc_init = 0;
  static int *psum_g,*listid_g;
  int ns = (nc+NTHREADS_RADIX-1)/NTHREADS_RADIX;

  if(n <= 0 || nc <= 0) {
    if(n_init > 0) {
      cudaFree(listid_g);
      cudaFree(psum_g);
    }
    n_init = 0;
    nc_init = 0;
  } else if(n > n_init || nc > nc_init) {
    if(n_init > 0) {
      cudaFree(listid_g);
      cudaFree(psum_g);
    }
    cudaMalloc((void **) &psum_g,sizeof(int) * ns);
    cudaMalloc((void **) &listid_g,sizeof(int) * n);
    n_init = n;
    nc_init = nc;
  }

  if(n > 0 && nc > 0) {
    /*
    int *listid = (int *) malloc(sizeof(int) * n);
    int *count = (int *) malloc(sizeof(int) * nc);
    int *count2 = (int *) malloc(sizeof(int) * nc);
    int *psum = (int *) malloc(sizeof(int) * ns);
    int *psum2 = (int *) malloc(sizeof(int) * ns);
    int i,s;
    */
    boxsum_stage1<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(nc,count_g);
    /*
    cudaThreadSynchronize();
    s = 0;
    for(i = 0; i<nc; i++)
      s += abs(count[i]);
    if(s != 0)
      printf("count not zeroed, s=%d\n",s);
    */

    boxsum_stage2<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(n,xin_g,listid_g,count_g);
    /*
    cudaThreadSynchronize();

    cudaMemcpy(listid,listid_g,sizeof(int) * n,cudaMemcpyDeviceToHost);
    cudaMemcpy(count,count_g,sizeof(int) * nc,cudaMemcpyDeviceToHost);
    s = 0;
    for(i = 0; i<nc; i++)
      s += count[i];
    if(s != n)
      printf("Error in count, s=%d, n=%d\n",s,n);
    */
    boxsum_stage3<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(nc,count_g,psum_g);
    /*
    cudaThreadSynchronize();

    cudaMemcpy(psum,psum_g,sizeof(int) * ns,cudaMemcpyDeviceToHost);

    for(i = 0; i<ns; i++) {
      int j;
      s = 0;
      for(j = 0; j<NTHREADS_RADIX; j++)
	if(i*NTHREADS_RADIX+j < nc) s += count[i*NTHREADS_RADIX+j];
      if(s != psum[i])
	printf("psum error, i=%d ns=%d s=%d psum=%d\n",i,ns,s,psum[i]);
    }
    */
    boxsum_stage4<<<1,NTHREADS_RADIX>>>(ns,psum_g);
    /*
    cudaThreadSynchronize();

    cudaMemcpy(psum2,psum_g,sizeof(int) * ns,cudaMemcpyDeviceToHost);

    s = 0;
    for(i = 0; i<ns; i++) {
      s += psum[i];
      if(s != psum2[i])
	printf("cumsum error in psum: s=%d psum2=%d i=%d ns=%d\n",
	       s,psum2[i],i,ns);
    }
    */
    boxsum_stage5<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(nc,count_g,psum_g);
    /*
    cudaThreadSynchronize();

    cudaMemcpy(count2,count_g,sizeof(int) * nc,cudaMemcpyDeviceToHost);

    s = 0;
    for(i = 0; i<nc; i++) {
      s += count[i];
      if(s != count2[i])
	printf("cumsum error in count: s=%d count2=%d i=%d nc=%d\n",
	       s,count2[i],i,nc);
    }
    */
    boxsum_stage6<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(n,xin_g,listid_g,count_g);
    boxsum_stage7<<<NBLOCKS_RADIX,NTHREADS_RADIX>>>(n,listid_g,xin_g,
						    data1in_g,data2in_g,xout_g,
						    data1out_g,data2out_g);
  }
}
