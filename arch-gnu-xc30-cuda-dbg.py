#!/usr/bin/env python
import os
# make sure to load the correct modules 
#  module unload PrgEnv-cray && module load PrgEnv-gnu && module load cudatoolkit
#

configure_options = [
  '--configModules=PETSc.Configure',
  '--optionsModule=config.compilerOptions ',
  '--CC=mpiicc ',
#  '--CPPFLAGS="-DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX" ',
  '--CXX=mpicxx',
  '--FC=mpiifort',
  '--download-chaco',
  '--download-hdf5',
  '--download-hypre',
  '--download-metis',
  '--download-ml',
  '--download-mumps',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-pastix',
  '--download-spai',
  '--download-suitesparse',
  '--download-superlu',
  '--download-superlu_dist',
  '--known-bits-per-byte=8',
  '--known-level1-dcache-assoc=8',
  '--known-level1-dcache-linesize=64',
  '--known-level1-dcache-size=32768',
  '--known-memcmp-ok=1',
  '--known-mpi-c-double-complex=1',
  '--known-mpi-long-double=1',
  '--known-mpi-shared-libraries=1',
  '--known-sizeof-MPI_Comm=4',
  '--known-sizeof-MPI_Fint=4',
  '--known-sizeof-char=1',
  '--known-sizeof-double=8',
  '--known-sizeof-float=4',
  '--known-sizeof-int=4',
  '--known-sizeof-long-long=8',
  '--known-sizeof-long=8',
  '--known-sizeof-short=2',
  '--known-sizeof-size_t=8',
  '--known-sizeof-void-p=8',
#  '--with-blas-lapack-include=/apps/all/imkl/11.3.1.150-iimpi-2016.01-GCC-4.9.3-2.25/mkl/include ',
#  '--with-blas-lapack-lib="-L/apps/all/imkl/11.3.1.150-iimpi-2016.01-GCC-4.9.3-2.25/mkl/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm" ',
  '--with-c2html=0 ',
  '--with-cxx-dialect=C++11 ',
  '--with-debugging=1 ',
  '--with-fortran ',
  '--with-mpi ',
  '--with-mpiexec=/apps/mpi/MPICH/3.2-GCC-4.9.3-2.25/bin/mpiexec',
  '--with-pic ',
#  '--with-scalapack-include=/apps/all/imkl/11.3.1.150-iimpi-2016.01-GCC-4.9.3-2.25/mkl/include ',
#  '--with-scalapack-lib="-L/apps/all/imkl/11.3.1.150-iimpi-2016.01-GCC-4.9.3-2.25/mkl/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm" ',
  '--with-shared-libraries ',
  '--with-valgrind ',
  '--with-windows-graphics=0 ',
  '--with-x ',
  'PETSC_ARCH=arch-gnu-xc30-cuda-dbg',
  
  # cuda fun
  '--with-cuda=1',
  '--with-cuda-arch=sm_35',
  '--with-cuda-dir=/apps/all/CUDA/7.5/',
  '--with-cudac=nvcc',
  '--CUDAFLAGS=-I/apps/mpi/MPICH/3.2-GCC-4.9.3-2.25/include/',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
