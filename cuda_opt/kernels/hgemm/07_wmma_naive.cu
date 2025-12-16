// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 00:53:54 on Mon, Feb 13, 2023
//
// Description: wmma naive hgemm
#include "cuda_fp16.h"
#include "mma.h"
#include "cublas_v2.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

using namespace nvcuda;

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void wmmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        // wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, half *d_A, int lda,
              half *d_B, int ldb, half *d_C, int ldc) {

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(n, WMMA_N), div_ceil(m, WMMA_M));

    wmmaNaiveKernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

}