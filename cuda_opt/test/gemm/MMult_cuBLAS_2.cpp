#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions

void CUBLAS_MMult(cublasHandle_t handle, int m, int n, int k, float *h_A, int lda,
              float *h_B, int ldb, float *h_C, int ldc) {
  // memcpy from host to device
  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_C, m * n * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif

checkCudaErrors(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    (void*)(&alpha), d_B, CUDA_R_32F, n, d_A, CUDA_R_32F, k,
    (void*)(&beta), d_C, CUDA_R_32F, n, compute_type, CUBLAS_GEMM_DEFAULT));
// memcpy from device to host
  checkCudaErrors(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  // free device memory
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

}

void CUBLAS_MMult(cublasHandle_t handle, int m, int n, int k,
                       half* a, int lda,
                       half* b, int ldb,
                       half* c, int ldc) {
    // memcpy from host to device
    half *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**)&d_a, m * k * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&d_b, k * n * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&d_c, m * n * sizeof(half)));
    checkCudaErrors(cudaMemcpy(d_a, a, m * k * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, k * n * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c, c, m * n * sizeof(half), cudaMemcpyHostToDevice));
    
    const half alpha = __float2half(1.0f);
    const half beta  = __float2half(0.0f);

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 d_b, CUDA_R_16F, ldb,
                 d_a, CUDA_R_16F, lda,
                 &beta,
                 d_c, CUDA_R_16F, ldc,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    // memcpy from device to host
    checkCudaErrors(cudaMemcpy(c, d_c, m * n * sizeof(half), cudaMemcpyDeviceToHost));
    // free device memory
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

}