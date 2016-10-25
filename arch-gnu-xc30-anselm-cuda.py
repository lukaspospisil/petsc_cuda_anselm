#!/usr/bin/env python
import os
# make sure to load the correct modules 
#  module unload PrgEnv-cray && module load PrgEnv-gnu && module load cudatoolkit
#

# Get CUDATOOLKIT_HOME from environment
CUDATOOLKIT_HOME=os.getenv('CUDATOOLKIT_HOME')
if not CUDATOOLKIT_HOME :
  raise Exception("CUDATOOLKIT_HOME not defined in the environment. Did you forget to load modules?")

# Get MPICH DIR from the environment (see note below about nvcc)
MPICH_DIR=os.getenv('MPICH_DIR')

configure_options = [
# On cray cc,CC,ftn are eqivalent to mpicc,mpiCC,mpif90
# Note that we add some flags, OVERRIDING any existing COPTFLAGS and CXXOPTFLAGS..
  '--with-cc=cc',
  '--with-cxx=mpicxx',
#  '--with-fc=gfortran',
  '--with-fc=0',

  'COPTFLAGS=-O3',
  'CXXOPTFLAGS=-03',
#  'FOPTFLAGS=-03',

  '--with-clib-autodetect=0',
  '--with-cxxlib-autodetect=0',
#  '--with-fortranlib-autodetect=0',

  '--with-shared-libraries=0',
#  '--with-debugging=0',
  '--with-valgrind=0',

  # Note: this is a batch system, but under the assumption that
  #       the login nodes are indeed identical to the compute nodes
  #       (not always true in practice!) then we shouldn't need this
  #'--with-batch',

  '--known-mpi-shared-libraries=1',

  '--with-x=0',
  #'--with-hwloc=0'

  'PETSC_ARCH=arch-gnu-xc30-anselm-cuda',
  #'--with-blas-lapack-lib=-L/opt/cray...'


  '--with-cuda=1',
  '--with-cuda-arch=sm_35',
  '--with-cuda-dir='+CUDATOOLKIT_HOME,
  '--with-cudac=nvcc',

  # The cc/CC/ftn compiler wrappers provide
  # things like MPI headers, but nvcc
  # doesn't have them, so we add them
  # manually here:
  '--CUDAFLAGS=-I'+MPICH_DIR+'/include/',
  
  '--with-batch=1',
  '--download-f2cblaslapack=1',


  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
