#include "computations.h"


void test_computations(void)
{
  printf("Hello from computation.c\n");
}

void parallel_lanczos_algo(int global_nolines, int *counts, int *displs, int nocols, int m, double *a, double *v, double *w, double *alpha, double *beta)
{
  int size, rank;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  const int srow = displs[rank]; // start row
  const int local_nolines = counts[rank];
  
  double oobeta = 1.0;
  int incv = 1, incw = 1;
  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  int lda  = nocols;

  double local_alpha, local_beta;

  for(int j=1; j<m; j++)
  {
    // step 1: Very similar to the sequential version, we just need to adapt the global_nolines with local_nolines
    cblas_dcopy(local_nolines, v+(j-1)*global_nolines+srow, incv, w+j*local_nolines, incw);
    cblas_dgemv(layout, transa, local_nolines, nocols, 1.0, a, lda, v+j*nocols+srow, incv, -1.0*(*(beta+j)), w+j*local_nolines, incw);

    // step 2: Each proc prforms a part of the dot product, the MPI_ALLreduce is used to sum up all the partial sums and bcast on all the procs
    local_alpha = cblas_ddot(local_nolines, w+j*local_nolines, incw, v+j*global_nolines+srow, incv);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&local_alpha, alpha+j, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // step 3: daxpy is performed only on part of the vector
    cblas_daxpy(local_nolines, -1.0*(*(alpha+j)), v+j*global_nolines+srow, incv, w+j*local_nolines, incw);

    // step 4: the norm is not computed directly (dnrm2) is not used. The partial dot product is computed instead on each proc then MPI_Allreduce is used to sum
    local_beta = cblas_ddot(local_nolines, w+j*local_nolines, incw, w+j*local_nolines, incv);
    MPI_Allreduce(&local_alpha, beta+j+1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *(beta+j+1) = sqrt(*(beta+j+1));

    // step 5: Similar to the sequential version, v(j+1) is reconstructed at the end using MPI_Allgather
    oobeta =1.0/(*(beta+j+1));
    cblas_dcopy(local_nolines, w+j*local_nolines, incw, v+(j+1)*global_nolines+srow, incv);
    cblas_dscal(local_nolines, oobeta, v+(j+1)*global_nolines+srow, incv);
    MPI_Allgatherv(v+(j+1)*global_nolines+srow, local_nolines, MPI_DOUBLE, v+(j+1)+global_nolines, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
  }
}

void lanczos_algorithm(int nolines, int nocols, int m, double *a, double *v, double *w, double *alpha, double *beta)
{
  double oobeta = 1.0;
  int incv = 1, incw = 1;
  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  int lda  = nocols;

  for(int j = 1; j<m;j++)
  {
    /*
    the algorithm implemented is the one figure 8 page 13 (document Calcul des valeurs propresby Bernard PHILIPPE and Yousef SAAD in res dir)
    step 1 : w(j)= Av(j)-beta(j)v(j-1)
    step 2 : alpha(j) = (w(j)^T,v(j))
    step 3 : w(j) = w(j)-alpha(j)*v(j)
    step 4 : beta(j+1) = ||w(j)||_2
    step 5 : v(j+1) = w(j)/beta(j+1)
    */
    //printf("--------- iteration : %d -----------\n",j);

    //step 1

    /*
    since dgemv computes y<- alpha*Ax + beta*y
    w(j) = y here
    and v(j) = x
    alpha = 1.0
    for v(j-1) needs to appear, it is copied to w(j) which value will be overwritten by the calculation making it correct
    */
    cblas_dcopy(nolines, v+(j-1)*nolines, incv, w+j*nolines, incw);
    //printf("v(%d) copied in w(%d) by cblas_dcopy\n",j-1,j);
    //printvec(w+j*nolines, nolines, "w(j)");
    //printvec(v+j*nolines, nocols, "v(j)");

    cblas_dgemv(layout, transa, nolines, nocols, 1.0, a, lda, v+j*nocols, incv, -1.0*(*(beta+j)), w+j*nolines, incw);
    //printf("w is the output of cblas_dgemv(w(%d) = Av(%d)-beta(%d)*v(%d)\n",j,j,j,j-1);
    //printvec(w+j*nolines, nolines, "w");

    //step 2

    //printf("dotprod between w(%d) and v(%d)\n",j,j);
    *(alpha+j) = cblas_ddot(nolines, w+j*nolines, incw, v+j*nolines, incv);
    //printf("w(%d).v(%d) = %lf\n\n",j,j, *(alpha+j));

    //step 3

    cblas_daxpy(nolines, -1.0*(*(alpha+j)), v+j*nolines, incv, w+j*nolines, incw);
    //printf("w(%d) = w(%d)-alpha(%d)*v(%d)\n",j,j,j,j);
    //printvec(w+j*nolines, nolines, "w");

    //step 4

    *(beta+j+1) = cblas_dnrm2(nolines,w+j*nolines,incw);
    //printf("beta(%d) = ||v(%d)|| = %lf\n", j+1,j,*(beta+j+1));

    //step 5

    oobeta =1.0/(*(beta+j+1));// dscal is alpha*vector so 1/beta is needed (oobeta means one over beta)
    cblas_dcopy(nolines, w+j*nolines, incw, v+(j+1)*nolines, incv); // w(j) is written v(j+1)
    //printf("w(%d) copied in v(%d) by cblas_dcopy\n",j,j+1);
    //printvec(w+j*nolines, nolines, "w");

    cblas_dscal(nolines, oobeta, v+(j+1)*nolines, incv);//computes v(j+1) = alpha*v(j+1) = (1/beta(j+1))*v(j+1) [oobeta = aplha]= (1/beta(j+1))*w(j) [w(j) copied into v(j+1)]
    //printf("v(%d)/beta(%d) \n",j+1,j);
    //printvec(v+(j+1)*nolines, nolines, "v(j+1)");
  }
  MPI_Finalize();
}
