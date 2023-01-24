#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <lapacke.h>
#include <mpi.h>


void test_func(void);
void printvec(double * vec, int length, char *name);
void printmat(double * vec, int nolines, int nocols, char *name, CBLAS_LAYOUT layout);
double *A3(int n, CBLAS_LAYOUT layout, int srow, int erow);
double *A9(double a, double b, int n, CBLAS_LAYOUT layout, int srow, int erow);
double *AMn(int n, CBLAS_LAYOUT layout, int srow, int erow);
double *A1(void);
double *randsym (int n);
double *randunitvec (int n);
double mean (double *sample, int noelts);
double std (double *sample, int noelts, double mval);

void distribute_on_procs(int nolines, int *counts, int *displs);
