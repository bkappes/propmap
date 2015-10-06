#!/usr/bin/env python
"""
FILL IN THIS HERE
"""

import inspect
import os
import sys

if len(sys.argv) != 3:
    print >>sys.stderr, "Usage: %s file.in out.chgcar" % sys.argv[0]
    sys.exit(0)

import numpy as np
from numpy.linalg import norm
from ase.io import read, write
from ase.data import covalent_radii
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

infile = sys.argv[1]
chgcar = sys.argv[2]

# read in atoms
if infile.endswith('.vasp') or infile.endswith('.poscar'):
    atoms = read(infile, format='vasp')
else:
    atoms = read(infile)
natoms = np.int32(atoms.get_number_of_atoms())
# periodicity
pv = np.asarray(atoms.get_cell(), dtype=np.float32)
pv = np.reshape(pv, (pv.size,))
# number of divisions
resolution = 0.1
ndiv = np.asarray(np.ceil([norm(v)/resolution for v in atoms.get_cell()]), dtype=np.int32)
# scaled atom positions
spos = np.asarray(atoms.get_scaled_positions(), dtype=np.float32)
spos = np.reshape(spos, (spos.size,))
# atomic radii
radii = np.asarray([covalent_radii[z] for z in atoms.numbers], dtype=np.float32)
# set up grid
ngrid = np.int32(np.prod(ndiv))
grid = np.ndarray((ngrid,), dtype=np.float32)
grid.fill(np.finfo(np.float32).max)

# -----------------------

try:
    device_id = int(os.environ['CUDA_DEVICE'])
except KeyError:
    device_id = 0
device = cuda.Device(device_id)

DIM = 3
# number of threads that one block can handle
THREADS_PER_BLOCK = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
# number of blocks required to handle all atoms
N_BLOCKS = int(np.ceil(float(atoms.get_number_of_atoms())/THREADS_PER_BLOCK))
# maximum number of threads per block
N_THREAD = int(np.ceil(float(atoms.get_number_of_atoms())/N_BLOCKS))

path=os.path.dirname(os.path.realpath(inspect.getfile(inspect.currentframe())))
# ------------------------
# just-in-time compilation
# ------------------------
#with open(path + '/distance_BpV.cu') as ifs:
    #kernel_text = ifs.read()
#kernel = SourceModule(kernel_text)
# ------------------------
# precompiled kernel
# ------------------------
kernel = path + '/distance_BpV.cubin'
kernel = cuda.module_from_file(path + '/distance_BpV.cubin')

distance = kernel.get_function('distance_BpV')

shared_mem = DIM * N_THREAD * np.float32().nbytes
cuda_grid = (int(ngrid), N_BLOCKS)
cuda_blocks = (N_THREAD, 1, 1)

distance(cuda.InOut(grid), ngrid,
         cuda.In(pv),
         cuda.In(ndiv),
         cuda.In(spos),
         cuda.In(radii),
         natoms,
         block=cuda_blocks, grid=cuda_grid, shared=shared_mem)

# -----------------------

write(chgcar, atoms, format='vasp', direct=True, vasp5=True)
with open(chgcar, 'a', buffering=1048576) as ofs:
    ofs.write('\n')
    ofs.write(' '.join([str(n) for n in ndiv]) + '\n')
    count = 1
    for val in grid:
        ofs.write('{:.12f}'.format(float(val)))
        ofs.write(' ' if count % 5 else '\n')
        count += 1

