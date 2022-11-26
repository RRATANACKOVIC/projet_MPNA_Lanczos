#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

#include"../inc/func.h"

int main (void)
{
  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  printf(" Welcome on the Lanczos project !\n");
  test_func();
  int nolines = 3, nocols = 2;
  int lda  = nocols, incx = 1, incy = 1, incz = 1;
  double alpha = 1.0, beta = 1.0, oobeta = 1.0;
  double *a = (double*)calloc(nolines*nocols,sizeof(double));
  //first line
  *a = 1.0;
  *(a+1) = -1.0;
  //second line
  *(a+2) = -1.0;
  *(a+3) = 1.0;
  //third line
  *(a+4) = 1.0;
  *(a+5) = -1.0;
  printmat(a, nolines, nocols, "a", layout);
  double *x = (double*)calloc(nocols,sizeof(double));
  *x = -2.0;
  *(x+1) = 3.0;
  printvec(x, nocols, "x");
  double *y = (double*)calloc(nolines,sizeof(double));
  *y = 1.0;
  *(y+1) = 4.0;
  *(y+2) = 5.0;
  printvec(y, nolines, "y");
  double *z = (double*)calloc(nolines,sizeof(double));
  cblas_dcopy(nolines, y, incy, z, incz);
  printf("y copied in z by cblas_dcopy\n");
  printvec(z, nolines, "z");
  cblas_dgemv( layout, transa, nolines, nocols, alpha, a, lda, x, incx, beta, z, incz );
  printf("z is the output of cblas_dgemv\n");
  printvec(z, nolines, "z");
  printf("dotprod between y and z\n");
  alpha = cblas_ddot(nolines, y, incy, z, incz);
  printf("y.z = %lf\n", beta);
  cblas_daxpy(nolines, -alpha, y, incy, z, incz);
  printf("z = z-alpha*y\n");
  printvec(z, nolines, "z");
  printf("z = z-alpha*y\n");
  beta = cblas_dnrm2(nolines,z,incz);
  printf("beta = ||x|| = %lf\n", beta);
  oobeta =1.0/beta;
  double * t = (double *)calloc(nolines,sizeof(double));
  cblas_dscal(nolines, oobeta, z, incz);
  printf("z/||z|| \n");
  printvec(z, nolines, "z");
  return 0;
}
