#include"../inc/func.h"

void test_func(void)
{
  printf("The func.c file seems to be correctly linked.\n");
}

void printvec(double * vec, int length, char *name)
{
  printf("%s in R^%d: \n", name, length);
  for(int i = 0; i<length; i++)
  {
    printf("%s[%d] = %lf\n",name,i,*(vec+i));
  }
  printf("\n");
}

void printmat(double * vec, int nolines, int nocols, char *name, CBLAS_LAYOUT layout)
{
  printf("%s in R^(%d*%d): \n", name, nolines, nocols);
  if (layout == CblasRowMajor)
  {
      for(int i = 0; i<nolines; i++)
      {
        for(int j = 0; j<nocols; j++)
        {
          printf("%lf ",*(vec + i*nocols + j));
        }
        printf("\n");
      }
  }
  else
  {
    printf("CblasColMajor not implemented, sorry. \n");
  }
  printf("\n");
}

/*
The following functions are named after the arrays shown in the
test array document stored in res.
*/

double * A3(int n, CBLAS_LAYOUT layout)
{
  double * output = (double *)calloc(n*n, sizeof(double));
  if (layout == CblasRowMajor)
  {
    for(int j = 0; j<n; j++)
    {
      for(int i = 0; i<j; i++)
      {
        *(output + i*n + j) = n + 1 - j;
      }
    }
    for(int j = 0; j<n-1; j++)
    {
      for(int i = j+1; i<n; i++)
      {
        *(output + i*n + j) = n + 1 - j;
      }
    }
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

double *A9(double a, double b, int n, CBLAS_LAYOUT layout)
{
  double * output = (double *)calloc(n*n, sizeof(double));
  if (layout == CblasRowMajor)
  {
    *output = a;
    *(output+1) = b;
    for(int i = 1; i<n-1; i++)
    {
      *(output+(n+1)*i-1) = b;
      *(output+(n+1)*i) = a;
      *(output+(n+1)*i+1) = b;
    }
    *(output+(n+1)*(n-1)-1) = b;
    *(output+(n+1)*(n-1)) = a;
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

double *AMn(int n, CBLAS_LAYOUT layout)
{
  double * output = (double *)calloc(n*n, sizeof(double));
  if (layout == CblasRowMajor)
  {
    *output = 1.0;
    *(output+1) = -0.1;
    for(int i = 1; i<n-1; i++)
    {
      *(output+(n+1)*i-1) = 0.1;
      *(output+(n+1)*i) = (double)(i+1);
      *(output+(n+1)*i+1) = -0.1;
    }
    *(output+(n+1)*(n-1)-1) = 0.1;
    *(output+(n+1)*(n-1)) = (double)(n);
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

double *A1(void)
{
  double * output = (double *)calloc(8*8, sizeof(double));
  double a = 11111111.0, b = 9090909.0, c = 10891089.0, d = 8910891.0, e = 11108889.0, f = 9089091.0, g = 10888911.0, h = 8909109.0;
  *output=a;*(output+1)=-b; *(output+2)=-c;*(output+3)=d;*(output+4)=-e;*(output+5)=f;*(output+6)=g;*(output+7)=-h;
  *(output+8)=-b;*(output+9)=a;*(output+10)=d;*(output+11)=-c;*(output+12)=f;*(output+13)=-e;*(output+14)=-h;*(output+15)=g;

  *(output+16)=-c;*(output+17)=d;*(output+18)=a;*(output+19)=-b;*(output+20)=g;*(output+21)=-h;*(output+22)=-e;*(output+23)=f;
  *(output+24)=d;*(output+25)=-c;*(output+26)=-b;*(output+27)=a;*(output+28)=-h;*(output+29)=g;*(output+30)=f;*(output+31)=-e;

  *(output+32)=-e;*(output+33)=f;*(output+34)=g;*(output+35)=-h;*(output+36)=a;*(output+37)=-b;*(output+38)=-c;*(output+39)=d;
  *(output+40)=f;*(output+41)=-e;*(output+42)=-h;*(output+43)=g;*(output+44)=-b;*(output+45)=a;*(output+46)=d;*(output+47)=-c;

  *(output+48)=g;*(output+49)=-h;*(output+50)=-e;*(output+51)=f;*(output+52)=-c;*(output+53)=d;*(output+54)=a;*(output+55)=-b;
  *(output+56)=-h;*(output+57)=g;*(output+58)=f;*(output+59)=-e;*(output+60)=d;*(output+61)=-c;*(output+62)=-b;*(output+63)=a;
  return output;
}

double *randsym (int n)
{
  int inco = 1, incc = 1;
  double *output = (double *)calloc(n*n, sizeof(double));
  double *copy = (double *)calloc(n*n, sizeof(double));
  srand(time(NULL));
  //int seed = rand()%100;
  int seed[4] = {rand()%100, rand()%100, rand()%100, 2*(rand()%100)+1};
  LAPACKE_dlarnv_work(2,seed, n*n, output);// creates random array  the 2 is for uniform values in [-1,1]
  cblas_dcopy(n*n, output, inco, copy, incc);
  for (int i = 0; i<n; i++)
  {
    for (int j = 0; j<n; j++)
    {
      *(output+i*n+j) += *(copy+j*n+i);// a + a^T is symmetric
    }
  }
  return output;
}

double *randunitvec (int n)
{
  int inco = 1;
  double *output = (double *)calloc(n, sizeof(double));
  srand(time(NULL));
  int seed[4] = {rand()%100, rand()%100, rand()%100, 2*(rand()%100)+1};
  LAPACKE_dlarnv_work(2,seed, n, output);// creates random array  the 2 is for uniform values in [-1,1]
  cblas_dscal(n, 1.0/cblas_dnrm2(n, output,inco), output, inco);//computes v(j+1) = alpha*v(j+1) = (1/beta(j+1))*v(j+1) [oobeta = aplha]= (1/beta(j+1))*w(j) [w(j) copied into v(j+1)]

  return output;
}
