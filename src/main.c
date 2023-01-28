#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <lapacke.h>
#include <lapacke_utils.h>
#include "../inc/func.h"
#include "../inc/computations.h"

#include <mpi.h>

#define MASTER_RANK 0

int main (int argc, char **argv)
{
  // MPI init
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2)
  {
    printf("No output file mentionned, abort\n");
    exit(0);
  }

  char *filename = argv[1];

  CBLAS_LAYOUT layout = CblasRowMajor;
  /*
  ///////// layout \\\\\\\\\\

      | a b c |
 A =  | d e f |
      | g h i |
  A in CblasRowMajor : A = [a,b,c,d,e,f,g,h,i]
  A in CblasColMajor : A = [a,d,g,b,e,h,c,f,i]
  */
  CBLAS_TRANSPOSE transa = CblasNoTrans;// No transpose is used
 
  int n0 = 4,nmax = 5, nstep = 2, m0 = 1, mstep = 1, nrep = 10;
  nmax++;
  double mval = 0.0, stdval = 0.0;
  struct timespec start, end;
  double *duration = (double *)calloc(nrep,sizeof(double));

  FILE *fptr;
  if (rank == MASTER_RANK)
  {
    fptr = fopen(filename,"w");
    fprintf(fptr,"n ; m ; mean duration (ns) ; std (ns) \n");
    fclose(fptr);
  }

  for(int n = n0; n< nmax; n+=nstep)// nmax/nstep repetions for different
  {
    //printf("* n = %d\n",n);
    for(int m = m0; m<n+1; m+=mstep)
    {
      //printf(" - m = %d\n",m);
      for(int rep = 0; rep<nrep; rep++)// the m-th degree of Lanczos algorithm for size n problem is repeated nrep times to get statistics (mean and std)
      {
        //printf("  _ rep = %d\n",rep);
        double *alpha = (double *)calloc(m+1,sizeof(double));// stores the values of alpha(1), ..., alpha(m+1)
        double *beta = (double *)calloc(m+1,sizeof(double)); // stores the values of beta(1), ..., beta(m+1)
        //double *a = randsym(n);// the input array
	int counts[size];
	int displs[size];
	distribute_on_procs(n, counts, displs);
	int srow = displs[rank];
	int erow = displs[rank]+counts[rank];
	double *a2 = A3(n, layout, srow, erow);
        double *v = (double*)calloc(n*(m+1),sizeof(double));// stores all v vectors computed throughout the process
        //double *v1 = randunitvec(n);
        double *v1 = (double *)calloc(n, sizeof(double));
        v1[0] = 0.218100; v1[1] = -0.557204; v1[2] = -0.569334; v[3] = -0.563751;
	MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&v1[0],n,MPI_DOUBLE,MASTER_RANK,MPI_COMM_WORLD);
        cblas_dcopy(n, v1, 1, v+n, 1);
        free(v1);
        //if (rank == 1)
        //{
        //  printf("rank %d/%d\n", rank,size);
        //  printmat(v,m+1,n,"v",CblasRowMajor);
        //}
        double *w = (double*)calloc(counts[rank]*(m+1),sizeof(double));

	// waiting for all the proc to measure the lanczos algorithm
	MPI_Barrier(MPI_COMM_WORLD);
        if(rank==MASTER_RANK) clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //lanczos_algorithm(n, n, m, a, v, w, alpha, beta);
        parallel_lanczos_algo(n, counts, displs, n, m, a2, v, w, alpha, beta);
	if(rank==MASTER_RANK)
	{
	  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
          *(duration+rep) = (double)(1000000000*(end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec));// since the duration is in ns the billions multiplies the seconds to get ns
          //printf("rep n°%d, duration = %lf\n",rep+1,*(duration+rep));
          fprintf(stderr, "alpha %e %e %e %e\n", alpha[0], alpha[1], alpha[2], alpha[3]);
          fprintf(stderr, "beta %e %e %e %e\n", beta[0], beta[1], beta[2], beta[3]);
        }

        free(alpha);
        free(beta);
        free(a2);
        free(v);
        free(w);
      }
      //printf("mean duration (ns) = %lf (+/-) %lf\n", mval, stdval);
      if (rank == MASTER_RANK)
      {
	mval = mean(duration, nrep);
        stdval = std(duration,nrep,mval);
        fptr = fopen(filename,"a");
        fprintf(fptr,"%d ; %d ; %lf ; %lf\n",n,m,mval,stdval);
        fclose(fptr);
      }

    }
  }
  
  //double *aaa = A3(5, layout);
  //printmat(aaa, 5, 5, "test",layout);
/* 
  fprintf(stderr, "rank %d/%d \n",rank,size);

  int counts[size];
  int displs[size];
  distribute_on_procs(4, counts, displs);
  int srow = displs[rank];
  int erow = displs[rank]+counts[rank];
  fprintf(stderr, "srow %d erow %d \n", srow, erow);
  double *a2 = A3(4, layout, srow, erow);
  printmat(a2, counts[rank], 4, "test", layout);
*/
  // ending MPI
  MPI_Finalize();
  return 0;
}
