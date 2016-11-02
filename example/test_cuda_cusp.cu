/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* if you are testing long vector then you want to turn this off */
#define PRINT_VECTOR_CONTENT 1

#ifdef USE_CUDA

/* this is neccesary for calling Vec{CUDA,CUSP}{Get,Restore}ArrayReadWrite */
#ifdef PETSC_HAVE_CUSP
  /* taken from src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h */
#if defined(CUSP_VERSION) && CUSP_VERSION >= 500
#include <cusp/blas/blas.h>
#else
#include <cusp/blas.h>
#endif
#define CUSPARRAY cusp::array1d<PetscScalar,cusp::device_memory>
PETSC_EXTERN PetscErrorCode VecCUSPGetArrayReadWrite(Vec v, CUSPARRAY **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayReadWrite(Vec v, CUSPARRAY **a);
#else
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayReadWrite(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayReadWrite(Vec v, PetscScalar **a);
#endif

/* cuda error check */ 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"\n\x1B[31mCUDA error:\x1B[0m %s %s \x1B[33m%d\x1B[0m\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* kernel called in this example */
__global__ void this_is_kernel(double *x_local_arr, int idx_start, int n_local){
	int i = blockIdx.x*blockDim.x + threadIdx.x; /* compute my id */

	if(i<n_local){ /* maybe we call more than n_local kernels */
		x_local_arr[i] = idx_start + i;
	}

	/* if i >= n_local then relax and do nothing */	
}

#endif


/* to deal with errors, call Petsc functions with TRY(fun); original idea from Permon (Vaclav Hapla) */
static PetscErrorCode ierr; 
#define TRY( f) {ierr = f; do {if (PetscUnlikely(ierr)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_IN_CXX,0);}} while(0);}

int main( int argc, char *argv[] )
{
	/* problem dimension (length of vector) */
	int n = 13;
	
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

	TRY( PetscPrintf(PETSC_COMM_WORLD,"This is Petsc-MPIVEC example.\n") );
	TRY( PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n") );
	
/* SAY HELLO TO WORLD - to check if everything is working */
	
	/* give info about MPI */
	int size, rank; /* size and rank of communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	TRY( PetscPrintf(MPI_COMM_WORLD,"- number of processors: %d\n",size) );
	TRY( PetscSynchronizedPrintf(MPI_COMM_WORLD," - hello from processor: %d\n",rank) );
	TRY( PetscSynchronizedFlush(MPI_COMM_WORLD,NULL) );
	TRY( PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n") );

/* CREATE GLOBAL VECTOR */

	/* create the data vector */
	Vec x_global; /* global data vector */
	TRY( VecCreate(PETSC_COMM_WORLD,&x_global) );
#ifdef USE_CUDA
#ifdef PETSC_HAVE_CUSP
	TRY( PetscPrintf(MPI_COMM_WORLD,"- using VECMPICUSP\n") );
	TRY( VecSetType(x_global, VECMPICUSP) );
#else
	TRY( PetscPrintf(MPI_COMM_WORLD,"- using VECMPICUDA\n") );
	TRY( VecSetType(x_global, VECMPICUDA) );
#endif
#else
	TRY( PetscPrintf(MPI_COMM_WORLD,"- using VECMPI\n") );
	TRY( VecSetType(x_global, VECMPI) );
#endif
	TRY( VecSetSizes(x_global,PETSC_DECIDE,n) );
	TRY( VecSetFromOptions(x_global) );

	/* set some random values - just for fun */
	PetscRandom rnd; /* random generator */
	TRY( PetscRandomCreate(PETSC_COMM_WORLD,&rnd) );
	TRY( PetscRandomSetType(rnd,PETSCRAND) );
	TRY( PetscRandomSetFromOptions(rnd) );
	TRY( PetscRandomSetSeed(rnd,30) );

	/* get ranges of decomposition */
	int idx_start, idx_end, n_local;
	TRY( VecGetOwnershipRange(x_global, &idx_start, &idx_end) );
	n_local = idx_end - idx_start;

	/* generate random data */
	TRY( VecSetRandom(x_global, rnd) );

	/* destroy the random generator */
	TRY( PetscRandomDestroy(&rnd) );

	/* maybe print the content of the global vector ? */
//	if(PRINT_VECTOR_CONTENT){
//		TRY( VecView(x_global, PETSC_VIEWER_STDOUT_WORLD) );
//	}

	/* compute some BLAS operations on vector */
	
	/* NORM_2 */
	double result;
	TRY( VecNorm(x_global, NORM_2, &result) );
	TRY( PetscPrintf(MPI_COMM_WORLD,"- test norm: %f\n",result) );

	/* Duplicate */
	Vec y;
	TRY( VecDuplicate(x_global,&y) );
	
	/* VecSet */
	TRY( VecSet(y, 1.0) );
	
	/* AXPY */
	TRY( VecAXPY(y, -3.1, x_global) );

	/* dot product */
	TRY( VecDot(y,x_global, &result) );
	TRY( PetscPrintf(MPI_COMM_WORLD,"- test dot: %f\n",result) );

	/* scale */
	TRY( VecScale(y, -result) );

	/* pointwisemult */	
	TRY( VecPointwiseMult(x_global, x_global, y) );

	/* destroy temp vector */
	TRY( VecDestroy(&y) );


	/* the test: x_global = [0,1,2,3,4,5,...,n-1] */
	/* get local array (for own kernels/OpenMP fun) */
	PetscScalar *x_local_arr;

#ifdef USE_CUDA
	TRY( PetscPrintf(MPI_COMM_WORLD,"- calling CUDA kernel\n") );
#ifdef PETSC_HAVE_CUSP
	CUSPARRAY *x_cusp;
	TRY( VecCUSPGetArrayReadWrite(x_global,&x_cusp) );
	x_local_arr = thrust::raw_pointer_cast(x_cusp->data());
#else
	TRY( VecCUDAGetArrayReadWrite(x_global,&x_local_arr) );
#endif

	this_is_kernel<<<n_local, 1>>>(x_local_arr,idx_start,n_local); //TODO: compute optimal call
	gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */

#ifdef PETSC_HAVE_CUSP
	TRY( VecCUSPRestoreArrayReadWrite(x_global,&x_cusp) );
#else
	TRY( VecCUDARestoreArrayReadWrite(x_global,&x_local_arr) );
#endif
#else
	TRY( PetscPrintf(MPI_COMM_WORLD,"- calling OpenMP parfor\n") );
	TRY( VecGetArray(x_global,&x_local_arr) );

	#pragma omp parallel for
	for(int i = 0; i < n_local; i++){
		x_local_arr[i] = idx_start + i;
	}

	TRY( VecRestoreArray(x_global,&x_local_arr) );

#endif


	/* maybe print the content of the global vector ? */
	if(PRINT_VECTOR_CONTENT){
		TRY( VecView(x_global, PETSC_VIEWER_STDOUT_WORLD) );
	}

	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


