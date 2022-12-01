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
  int nolines = 4, nocols = 3;
  int lda  = nocols, incx = 1, incy = 1, incz = 1;
  double alpha = 1.0, beta = 1.0, oobeta = 1.0;
  double *a = (double*)calloc(nolines*nocols,sizeof(double));
  //first line
  *a = -1.0; *(a+1) = 0.0; *(a+2) = 1.0;
  //second line
  *(a+3) = 1.0;  *(a+4) = -1.0; *(a+5) = 0.0;
  //third line
  *(a+6) = 0.0;  *(a+7) = 1.0; *(a+8) = -1.0;
  //fourth line
  *(a+9) = 2.0;  *(a+10) = 0.0; *(a+11) = 1.0;
  printmat(a, nolines, nocols, "a", layout);
  double *x = (double*)calloc(nocols,sizeof(double));
  *x = 2.0;*(x+1) = -3.0;*(x+2) = 4.0;
  printvec(x, nocols, "x");
  double *y = (double*)calloc(nolines,sizeof(double));
  *y = 1.0; *(y+1) = 2.0; *(y+2) = 3.0; *(y+3) = 4.0;
  printvec(y, nolines, "y");
  double *z = (double*)calloc(nolines,sizeof(double));
  cblas_dcopy(nolines, y, incy, z, incz);
  printf("y copied in z by cblas_dcopy\n");
  printvec(z, nolines, "z");
  cblas_dgemv( layout, transa, nolines, nocols, alpha, a, lda, x, incx, beta, z, incz );
  printf("z is the output of cblas_dgemv(y = Ax-y)\n");
  printvec(z, nolines, "z");
  printf("dotprod between y and z\n");
  alpha = cblas_ddot(nolines, y, incy, z, incz);
  printf("y.z = %lf\n", alpha);
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
  double *a3 = A3(3, layout);
  printmat(a3, 3, 3, "A3", layout);
  double *a9 = A9(5.0, 2.0,10, layout);
  printmat(a9, 10, 10, "A9", layout);
  double *amn = AMn(5, layout);
  printmat(amn, 5, 5, "AMn", layout);
  double *a1 = A1();
  printmat(a1, 8, 8, "A1", layout);
  return 0;
}
