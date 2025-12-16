#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
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
        As[threadIdx.y * BK + threadIdx.x] = A[threadIdx.y * K + threaxIdx.x];
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
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程总共可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int j = 0; j < TM; j++) {
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}

// template instantiation declaration
// template __global__ void sgemm_v4<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

int main(){
    const int N = 1024;
    const int M = 1024;
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
    
    //一个block处理tileC (BM,BN)
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    dim3 blockDim(BM, BN);
    dim3 gridDim(CEIL(M, BM), CEIL(N, BN));

    gemm_threadblock_tile<BM,BN,BK><<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, K);
    
}


