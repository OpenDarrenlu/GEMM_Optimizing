#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <utils.cuh>
#define CEIL(a, b) ((a + b - 1) / b)
#define warpsize 32

template<int BM, int BN, int BK>
__global__ void gemm_block_tile(float *A, float *B, float *C, int N, int M, int K){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx >= M || idy >= N) return;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];


    //移动到的当前的block
    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];


    float tmp = 0.0f;
    for(int k = 0 ;k  < K; k += BK){
        //缓存A-tile 和B-tile
        As[threadIdx.y * BK + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        Bs[threadIdx.x * BN + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        __syncthreads();
        A += BK;
        B += BK * N;
        //计算C-tile
        for(int i = 0 ; i < BK ;i++){
            tmp += As[threadIdx.y * BK + i] * Bs[threadIdx.x + i * BN];
        }
        __syncthreads();
    }
    C[threadIdx.y * N + threadIdx.x] = tmp;
}



template<const int BLOCK_SIZE>
__global__ void sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // blockId and threadId
    int bx = blockIdx.x;
    int by = blockIdx.y;    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    // NVIDIA GeForce GTX 1050's sharedMemPerBlock is 48KB = 48*1024B = 49152B(0xc000)
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 48*1024/4 = 12288
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();  // 同步同一个线程块(block)中的线程，执行到同一个点
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

// template instantiation declaration


// template __global__ void sgemm_v4<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
// Grid：(CEIL(N, BN), CEIL(M, BM))
// Block：(CEIL(BN, TN), CEIL(BM, TM))
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    int tm = threadIdx.y;
    int tn = threadIdx.x;


    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[bm * BM * K];
    B = &B[bn * BN];
    C = &C[bm * BM * N + bn * BN];
    for (int k = 0; k < K; k += BK) {
        // 缓存As 和 Bs
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < BK*TN/BN; j++) {
            }
        }
        // 同步所有线程缓存完成
        __syncthreads();  // 同步同一个线程块(block)中的线程，执行到同一个点
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp[tm][tn] += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    for ()

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；

}

// template instantiation declaration
// template __global__ void sgemm_v4<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

int main(){
    const int N = 1024*32;
    const int M = 1024*32;
    const int K = 128;
    const int repeat = 10;
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(N*K*sizeof(float));
    h_B = (float*)malloc(K*M*sizeof(float));        
    h_C = (float*)malloc(N*M*sizeof(float));
    for(int i = 0; i < N*K; i++){
        h_A[i] = 1.0f;
    }
    for(int i = 0; i < K*M; i++){
        h_B[i] = 1.0f; 
    }
    for(int i = 0; i < N*M; i++){
        h_C[i] = 0.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*K*sizeof(float));
    cudaMalloc(&d_B, K*M*sizeof(float));        
    cudaMalloc(&d_C, N*M*sizeof(float));
    cudaMemcpy(d_A, h_A, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N*M*sizeof(float), cudaMemcpyHostToDevice);
    
    //test sgemm_v4<128, 128, 8, 8, 8>
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 blockDim(CEIL(BN, TN), CEIL(BM, TM));
    dim3 gridDim( CEIL(N, BN), CEIL(M, BM));
    sgemm_v4<BM,BN,BK,TM,TN><<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    // test accuracy
    cudaMemcpy(h_C, d_C, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < N*M; i++){
        if(h_C[i] != K){
            printf("error at %d, expect %f, get %f\n", i, K*1.0, h_C[i]);
            break;
        }
    }
    printf("%f\n", TIME_RECORD(repeat, ([&]{sgemm_v4<BM,BN,BK,TM,TN><<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);})));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

}


