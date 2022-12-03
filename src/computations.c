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
}
