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
