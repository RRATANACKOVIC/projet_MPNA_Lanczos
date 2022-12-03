#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <lapacke.h>
#include <lapacke_utils.h>
#include "../inc/func.h"
#include "../inc/computations.h"

int main (void)
{
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
  int nolines = 4, nocols = 4, m = 4;
  int lda  = nocols, incv = 1, incw = 1, incz = 1;
  double oobeta = 1.0;//oobeta = one over beta, this variable's only purpose is to fit dscal's prototype
  double *alpha = (double *)calloc(m+1,sizeof(double));// stores the values of alpha(1), ..., alpha(m+1)
  double *beta = (double *)calloc(m+1,sizeof(double)); // stores the values of beta(1), ..., beta(m+1)
  double *a = (double*)calloc(nolines*nocols,sizeof(double));// the input array
  //first line
  *a = -1.0; *(a+1) = 1.0; *(a+2) = -1.0;*(a+3) = 1.0;
  //second line
  *(a+4) = 1.0; *(a+5) = -1.0;*(a+6) = 1.0;  *(a+7) = -1.0;
  //third line
  *(a+8) = -1.0;*(a+9) = 1.0;  *(a+10) = -1.0; *(a+11) = 1.0;
  //fourth line
  *(a+12) = 1.0; *(a+13) = -1.0;*(a+14) = 1.0;  *(a+15) = -1.0;
  printmat(a, nolines, nocols, "a", layout);
  double *v = (double*)calloc(nocols*(m+1),sizeof(double));// stores all v vectors computed throughout the process
  /*
  there is no layout v are following each other, v0 is full of zeros
  */
  *(v+nolines) = 2.0;*(v+nolines+1) = -3.0;*(v+nolines+2) = 4.0; *(v+nolines+3) = -5.0;
  printvec(v+nolines, nocols, "v");
  double *w = (double*)calloc(nolines*(m+1),sizeof(double));
  *w = 1.0; *(w+1) = 2.0; *(w+2) = 3.0; *(w+3) = 4.0;
  printvec(w, nolines, "w");
  if(nolines !=nocols)
  {
    printf("Matrix isn't square, abort.\n");
    exit(0);
  }

  lanczos_algorithm(nolines, nocols, m, a, v, w, alpha, beta);
  printvec(v,(m+1)*nolines,"test");
  double * sym = randsym(10);
  printmat(sym, 10, 10, "sym", layout);

  return 0;
}
