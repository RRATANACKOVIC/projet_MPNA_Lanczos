#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

#include <math.h>
#include <mpi.h>

#include "../inc/func.h"

void test_computations(void);
void lanczos_algorithm(int nolines, int nocols, int m, double *a, double *v, double *w, double *alpha, double *beta);
void parallel_lanczos_algo(int global_nolines, int *counts, int *displs, int nocols, int m, double *a, double *v, double *w, double *alpha, double *beta);
