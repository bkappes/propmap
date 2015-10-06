NVCC = /usr/local/cuda/bin/nvcc
#ARCH = -gencode 'arch=compute_20,code=sm_20'
ARCH = -gencode 'arch=compute_30,code=sm_30'
#ARCH = -gencode 'arch=compute_35,code=sm_35'
COMPILE = --cubin

.PHONY: clean distance

distance: distance_BpV.cu
	$(NVCC) $(ARCH) $(COMPILE) $<

clean:
	rm -rf *.cubin

