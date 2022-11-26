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
  int lda  = nolines, incx = 1, incy = 1;
  double alpha = 1.0, beta = 1.0;
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
  cblas_dgemv( layout, transa, nolines, nocols, alpha, a, lda, x, incx, beta, y, incy );
  printvec(y, nolines, "y");
  return 0;
}
