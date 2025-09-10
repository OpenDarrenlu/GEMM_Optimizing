/* Create macros so that the matrices are stored in row-major order */
#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

#include <cblas.h>
/* Routine for computing C = A * B + C */

// 泛型版本，支持 float 和 half
template <typename T>
void REF_MMult(int m, int n, int k, T *a, int lda, T *b, int ldb,
               T *c, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda,
              b, ldb, 0.0f, c, ldc);
}