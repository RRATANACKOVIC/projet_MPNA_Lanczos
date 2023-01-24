#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <mpi.h>

#include "func.h"

void test_computations(void);
void lanczos_algorithm(int nolines, int nocols, int m, double *a, double *v, double *w, double *alpha, double *beta);

