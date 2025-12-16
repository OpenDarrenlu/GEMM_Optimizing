#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
// 泛型版本，支持 float 和 half
template <typename T>
void CUBLAS_MMult(cublasHandle_t handle, int m, int n, int k, T *d_A, int lda,
              T *d_B, int ldb, T *d_C, int ldc) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, n, d_A, k, &beta, d_C, n));
}
