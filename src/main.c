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
  int nolines = 4, nocols = 4, m = 4;
  int lda  = nocols, incv = 1, incw = 1, incz = 1;
  double oobeta = 1.0;
  double *alpha = (double *)calloc(m+1,sizeof(double));
  double *beta = (double *)calloc(m+1,sizeof(double));
  double *a = (double*)calloc(nolines*nocols,sizeof(double));
  //first line
  *a = -1.0; *(a+1) = 1.0; *(a+2) = -1.0;*(a+3) = 1.0;
  //second line
  *(a+4) = 1.0; *(a+5) = -1.0;*(a+6) = 1.0;  *(a+7) = -1.0;
  //third line
  *(a+8) = -1.0;*(a+9) = 1.0;  *(a+10) = -1.0; *(a+11) = 1.0;
  //fourth line
  *(a+12) = 1.0; *(a+13) = -1.0;*(a+14) = 1.0;  *(a+15) = -1.0;
  printmat(a, nolines, nocols, "a", layout);
  double *v = (double*)calloc(nocols*(m+1),sizeof(double));
  *(v+nolines) = 2.0;*(v+nolines+1) = -3.0;*(v+nolines+2) = 4.0; *(v+nolines+3) = -5.0;
  printvec(v+nolines, nocols, "v");
  double *w = (double*)calloc(nolines*(m+1),sizeof(double));
  *w = 1.0; *(w+1) = 2.0; *(w+2) = 3.0; *(w+3) = 4.0;
  printvec(w, nolines, "w");
  if(nolines !=nocols)
  {
    printf("matrix isn't square, abort\n");
    exit(0);
  }
  for(int j = 1; j<m;j++)
  {
    printf("--------- iteration : %d -----------\n",j);

    cblas_dcopy(nolines, v+(j-1)*nolines, incv, w+j*nolines, incw);
    printf("v(j-1) copied in w(j) by cblas_dcopy\n");
    printvec(w+j*nolines, nolines, "w(j)");

    printvec(v+nolines, nocols, "v(j)");

    cblas_dgemv(layout, transa, nolines, nocols, 1.0, a, lda, v+j*nocols, incv, *(beta+j), w+j*nolines, incw);
    printf("w is the output of cblas_dgemv(w = Av-w)\n");
    printvec(w+j*nolines, nolines, "w");

    printf("dotprod between w and z\n");
    *(alpha+j) = cblas_ddot(nolines, w+j*nolines, incw, v+j*nolines, incv);
    printf("w(j).v(j) = %lf\n", *(alpha+j));

    cblas_daxpy(nolines, -1.0*(*(alpha+j)), v+j*nolines, incv, w+j*nolines, incw);
    printf("z = z-alpha*w\n");
    printvec(w+j*nolines, nolines, "w");

    printf("w = w-alpha*v\n");

    *(beta+j+1) = cblas_dnrm2(nolines,w+j*nolines,incw);
    printf("beta = ||v|| = %lf\n", *(beta+j+1));

    oobeta =1.0/(*(beta+j+1));
    cblas_dcopy(nolines, w+j*nolines, incw, v+(j+1)*nolines, incv);
    printf("v(j-1) copied in w(j) by cblas_dcopy\n");
    printvec(w+j*nolines, nolines, "w");

    cblas_dscal(nolines, oobeta, v+(j+1)*nolines, incv);
    printf("z/||z|| \n");
    printvec(v+(j+1)*nolines, nolines, "v(j+1)");
  }

  /*
  double *a3 = A3(3, layout);
  printmat(a3, 3, 3, "A3", layout);
  double *a9 = A9(5.0, 2.0,10, layout);
  printmat(a9, 10, 10, "A9", layout);
  double *amn = AMn(5, layout);
  printmat(amn, 5, 5, "AMn", layout);
  double *a1 = A1();
  printmat(a1, 8, 8, "A1", layout);
  */

  return 0;
}
