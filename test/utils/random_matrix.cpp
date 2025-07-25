#include <stdlib.h>

#define A(i, j) a[(i)*lda + (j)]

double drand48();
void random_matrix(int m, int n, float *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
#if 1
      A(i, j) = 2.0 * (float)drand48() - 1.0; // [-1, 1]
#else
      A(i, j) = j;
#endif
}
