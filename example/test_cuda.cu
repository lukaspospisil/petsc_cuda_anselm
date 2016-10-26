/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* if you are testing long vector then you want to turn this off */
#define PRINT_VECTOR_CONTENT 1

#ifdef USE_CUDA
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
	TRY( VecSetType(x_global, VECMPICUDA) );
#else
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
	for(i=0;i<n;i++){
		coeff[i]= 1/(double)(i+1);
	}
	TRY( VecMAXPY(y, -3.1, x_global) );

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
	/* get local array (for own kernels/OpenMPI fun) */
	double *x_local_arr;

#ifdef USE_CUDA
	TRY( VecCUDAGetArrayReadWrite(x_global,&x_local_arr) );

	this_is_kernel<<<n_local, 1>>>(x_local_arr,idx_start,n_local); //TODO: compute optimal call
	gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */

	TRY( VecCUDARestoreArrayReadWrite(x_local_global,&x_arr) );
#else
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


