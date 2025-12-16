#pragma _HELPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]
#define abs(x) ((x) < 0.0 ? -(x) : (x))

// 泛型版本，支持 float 和 half
template <typename T>
float compare_matrices(int m, int n, const T* a, int lda, const T* b, int ldb) {
    float max_diff = 0.0f;
    int printed = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float va = static_cast<float>(A(i, j)); // half 会自动转 float
            float vb = static_cast<float>(B(i, j));
            float diff = abs(va - vb);
            max_diff = (diff > max_diff ? diff : max_diff);

            if (printed == 0 && diff > 0.5f) {
                printf("\n error: i %d  j %d diff %f  got %f  expect %f ",
                       i, j, diff, va, vb);
                printed = 1;
            }
        }
    }
    return max_diff;
}

// 泛型版本，支持 float 和 half
template <typename T>
void copy_matrix(int m, int n, T *a, int lda, T *b, int ldb) {
  int i, j;

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++)
      B(i, j) = A(i, j);
}
// 泛型版本，支持 float 和 half
template <typename T>
void print_matrix(int m, int n, T *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f\t", A(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

double drand48();

// 泛型版本，支持 float 和 half
template <typename T>
void random_matrix(int m, int n, T *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
#if 1
      A(i, j) = 2.0 * drand48() - 1.0; // [-1, 1]
#else
      A(i, j) = j;
#endif
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
