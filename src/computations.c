#include "../inc/computations.h"

void test_computations(void)
{
  printf("Hello from computation.c\n");
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
    //printf("--------- iteration : %d -----------\n",j);

    cblas_dcopy(nolines, v+(j-1)*nolines, incv, w+j*nolines, incw);
    //printf("v(%d) copied in w(%d) by cblas_dcopy\n",j-1,j);
    //printvec(w+j*nolines, nolines, "w(j)");
    //printvec(v+j*nolines, nocols, "v(j)");

    cblas_dgemv(layout, transa, nolines, nocols, 1.0, a, lda, v+j*nocols, incv, *(beta+j), w+j*nolines, incw);
    //printf("w is the output of cblas_dgemv(w(%d) = Av(%d)-beta(%d)*v(%d)\n",j,j,j,j-1);
    //printvec(w+j*nolines, nolines, "w");

    //printf("dotprod between w(%d) and v(%d)\n",j,j);
    *(alpha+j) = cblas_ddot(nolines, w+j*nolines, incw, v+j*nolines, incv);
    //printf("w(%d).v(%d) = %lf\n\n",j,j, *(alpha+j));

    cblas_daxpy(nolines, -1.0*(*(alpha+j)), v+j*nolines, incv, w+j*nolines, incw);
    //printf("w(%d) = w(%d)-alpha(%d)*v(%d)\n",j,j,j,j);
    //printvec(w+j*nolines, nolines, "w");

    *(beta+j+1) = cblas_dnrm2(nolines,w+j*nolines,incw);
    //printf("beta(%d) = ||v(%d)|| = %lf\n", j+1,j,*(beta+j+1));

    oobeta =1.0/(*(beta+j+1));
    cblas_dcopy(nolines, w+j*nolines, incw, v+(j+1)*nolines, incv);
    //printf("w(%d) copied in v(%d) by cblas_dcopy\n",j,j+1);
    //printvec(w+j*nolines, nolines, "w");

    cblas_dscal(nolines, oobeta, v+(j+1)*nolines, incv);
    //printf("v(%d)/beta(%d) \n",j+1,j);
    //printvec(v+(j+1)*nolines, nolines, "v(j+1)");
  }
}
