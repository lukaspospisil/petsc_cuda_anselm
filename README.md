# petsc_cuda_anselm

see `load_modules` for bash stuff, see `arch-gnu-xc30-cuda-dbg.py` for petsc compilation stuff (call this python script from PETSc directory to configure with these options)

example could be compiled using cmake:
```
cd example
mkdir build
cd build
cmake -DFIND_PETSC=ON -DUSE_CUDA=ON ..
make
```
to test openmpi instead of CUDA, compile with `-DUSE_CUDA=OFF` and do not forget to set `OMP_NUM_THREADS`

for interactive mode, call `qsub example/pbs_script/pbs_interactive2.pbs`

in `load_modules` I defined an alias `pmpiexec` to call mpiexec from PETSc, it can be used in interactive mode
```
pmpiexec -n 2 ./test_cuda
```


