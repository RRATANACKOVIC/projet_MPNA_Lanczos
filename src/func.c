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
The following function distribute a matrix rows on multiple procs 
*/

void distribute_on_procs(int nolines, int *counts, int *displs)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  displs[0] = 0;
  for(int rank=0; rank<size; rank++)
  {
    if (rank < nolines%size)
    {
      srow = (nolines/size)*rank + rank;
      erow = (nolines/size)*(rank+1) + rank + 1;
    }
    else
    {
      srow = (nolines/size)*rank + nolines%size;
      erow = (nolines/size)*(rank+1) + nolines%size;
    }
    counts[rank] = erow-srow;
    if(rank>0) displs[rank] = displs[rank-1]+counts[rank-1];
  }
}

/*
The following functions are named after the arrays shown in the
test array document stored in res.
*/

// parallel version
// n: the number of both of rows and columns before distribution. It is also the number of columns avec after distribution
// layout
// norows: number of rows after distribution
double * A3(int n, CBLAS_LAYOUT layout, int srow, int erow)
{
  local_nolines = erow-srow
  double * output = (double *)calloc(n*local_nolines, sizeof(double));
  if (layout == CblasRowMajor)
  {
    for(int i=0; i<local_nolines; j++)
    {
      for(int j=0; j<n; j++)
      {
	if((i+srow)<=j)
	{
	  *(output+ i*n + j) = n + 1 - j;
	}
	else
	{
	  *(output+ i*n + j) = n + 1 - i - srow;
	}
      }
    }
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

/*
double * A3(int n, CBLAS_LAYOUT layout)
{
  double * output = (double *)calloc(n*n, sizeof(double));
  if (layout == CblasRowMajor)
  {
    for(int j = 0; j<n; j++)// 0 to n-1 instead of 1 to n
    {
      for(int i = 0; i<j+1; i++)// if i<j, the jth line is 0 so i<j+1 to avoid that
      {
        *(output + i*n + j) = n - j; // 0<j<n-1 and in the document 1<j<n
      }
    }
    for(int j = 0; j<n-1; j++)
    {
      for(int i = j+1; i<n; i++)
      {
        *(output + i*n + j) = n - i;
      }
    }
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}*/

double *A9(double a, double b, int n, CBLAS_LAYOUT layout, int srow, int erow)
{
  local_nolines = erow-srow
  double * output = (double *)calloc(n*local_nolines, sizeof(double));

  if (layout == CblasRowMajor)
  {
    int sline = 0;
    int eline = local_nolines;
    if (srow == 0)
    {
      *output = a;
      *(output+1) = b;
      sline = 1;
    }
    if (erow == n)
    {
      eline = local_nolines-1
      *(output+(n+1)*eline-1) = b;
      *(output+(n+1)*eline) = a;
    }
    for(int i=sline; i<eline; i++)
    {
      *(output+(n+1)*i+srow-1) = b;
      *(output+(n+1)*i+srow) = a;
      *(output+(n+1)*i+srow+1) = b;
    }
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

/*
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
}*/

double *AMn(int n, CBLAS_LAYOUT layout, int srow, int erow)
{
  local_nolines = erow-srow
  double * output = (double *)calloc(n*local_nolines, sizeof(double));

  if (layout == CblasRowMajor)
  {
    int sline = 0;
    int eline = local_nolines;
    if (srow == 0)
    {
      *output = 1.0;
      *(output+1) = -0.1;
      sline = 1;
    }
    if (erow == n)
    {
      eline = local_nolines-1
      *(output+(n+1)*eline-1) = 0.1;
      *(output+(n+1)*eline) = (double)n;
    }
    for(int i=sline; i<eline; i++)
    {
      *(output+(n+1)*i+srow-1) = 0.1;
      *(output+(n+1)*i+srow) = (double)(i+srow+1);
      *(output+(n+1)*i+srow+1) = -0.1;
    }
  }
  else
  {
    printf("CblasColMajor not implemented, the array is full of zeros \n");
  }
  return output;
}

/*
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
}*/

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

double *randunitvec (int n)// creates a vector of random values which euclidean's norm is equal to 1
{
  int inco = 1;
  double *output = (double *)calloc(n, sizeof(double));
  srand(time(NULL));
  int seed[4] = {rand()%100, rand()%100, rand()%100, 2*(rand()%100)+1};//the seed is used to generate the random vector
  LAPACKE_dlarnv_work(2,seed, n, output);// creates random array  the 2 is for uniform values in [-1,1]
  cblas_dscal(n, 1.0/cblas_dnrm2(n, output,inco), output, inco);//multiplies by 1/||output|| to have a norm equal to one

  return output;
}

double mean (double *sample, int noelts)
{
  double output = 0.0;
  for(int i = 0; i<noelts; i++)
  {
    output += *(sample+i);
  }
  output /= (double)(noelts);

  return output;
}

double std (double *sample, int noelts, double mval)// computes standard deviation
{
  return sqrt(cblas_ddot(noelts, sample, 1, sample, 1)/((double)(noelts))-mval*mval);//std = sum((xi-mean(x))²)= sum(xi²)-mean(x)²
}
