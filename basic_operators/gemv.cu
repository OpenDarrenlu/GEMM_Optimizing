#include <iostream>
#include <vector>
#include <random>
#include <cfloat> // For FLT_MAX

// CUDA 运行时API
#include <cuda_runtime.h>
#define CEIL(a, b) (a + b-1)/b
#define warp_size 32

// dim3 block_size(32);
// dim3 grid_size(m);
#define checkCudaErrors(func) {                                                   \
    cudaError_t e = (func);                                                       \
    if(e != cudaSuccess)                                                          \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));  \
}

// dim3 dimGrid(m);
// dim3 dimBlock(32);
// 适合K∈[32,128]使用，小于32或大于128有进一步的优化方法
//一个block只有一个warp
// My gemm Performance= 20.27 GFlop/s, Time= 0.003 msec, Size= 65536 Ops,
__global__ void gemv_v1(float* d_A, float *d_B, float *d_C, const int M, const int N){
    //每个block处理一行A和一列的
    int len_idx = threadIdx.x % warp_size;
    int row = blockIdx.x ;
    if(row > M) return;
    float val= 0.0f;
    float B = 0;

    int npartion = CEIL(N, warp_size); //每个线程处理的元素个数
    #pragma unroll
    for(int i = 0; i < npartion; i++){
        int col = i * warp_size + len_idx;
        B = d_B[col];
        // val += (col < N) ? (d_A[row * N + col] * d_B[col]):0.0f;
        val += (col < N) ? (d_A[row * N + col] * B):0.0f;
    }
    for(int offset = warp_size >>1 ; offset > 0; offset >>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if(len_idx ==0) d_C[row] = val;
}
//一个block有多个warp
//N=32
//dim3 dimGrid(CEIL(m, 4));
//dim3 dimBlock(4,32);
// My gemm Performance= 23.68 GFlop/s, Time= 0.003 msec, Size= 65536 Ops,
__global__ void gemv_v2(float *d_A, float* d_B, float *d_C, const int M, const int N){
    int len_idx = threadIdx.x % warp_size;
    // int cur_row = threadIdx.x * blockDim.x + threadIdx.y;
    int cur_row = blockIdx.x * blockDim.x + threadIdx.y;
    int npartion = CEIL(N, warp_size);
    float val= 0.0f;
    float B = 0;


    #pragma unroll
    for(int i = 0; i<npartion; i++){
        int cur_col = i * warp_size + len_idx;
        B = d_B[cur_col];
        val += (cur_col < N) ? (d_A[cur_row * N + cur_col] * B):0.0f;
    }

    for(int offset = warp_size >>1 ; offset > 0 ; offset>>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if(len_idx == 0) d_C[cur_row] = val;
}
//N=16  一个warp处理两行，若是处理一行，则有出现warp闲置情况
__global__ void gemv_v1(float* d_A, float *d_B, float *d_C, const int M, const int N){
    //每个block处理一行A和一列的
    int len_idx = threadIdx.x % warp_size;
    int row = blockIdx.x ;
    if(row > M) return;
    float val= 0.0f;
    float B = 0;

    int npartion = CEIL(N, warp_size); //每个线程处理的元素个数
    #pragma unroll
    for(int i = 0; i < npartion; i++){
        int col = i * warp_size + len_idx;
        B = d_B[col];
        // val += (col < N) ? (d_A[row * N + col] * d_B[col]):0.0f;
        val += (col < N) ? (d_A[row * N + col] * B):0.0f;
    }
    for(int offset = warp_size >>1 ; offset > 0; offset >>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if(len_idx ==0) d_C[row] = val;
}
//N=128 向量化
int main(){
    //A: m * n
    const int m = 1024;
    const int n = 32;
    float*A, *B, *C ;
    const int BLOCK_SIZE = 128;
    float msecTotal = 0;
    int iteration = 1000;
    double duration[2] = {0, 0};
    double GFLOPS[2] = {0, 0};
    double GFLOPs = 2.0 * m * 1 * n;

    A = (float*)malloc(sizeof(float) * m * n);
    B = (float*)malloc(sizeof(float) * n);
    C = (float*)malloc(sizeof(float) * m);

    for( int i = 0; i < m * n; i++ ) {
        A[i] = (float)i/n;
    }
    for(int j =0 ;j <n; j++){
        B[j] = 1;
    }

    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * n);
    cudaMalloc(&d_B, sizeof(float) * n);
    cudaMalloc(&d_C, sizeof(float) * m);


    cudaMemcpy(d_A , A, sizeof(float) * m * n , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B , B, sizeof(float) * n, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));


    for (int run = 0 ; run < iteration; run ++ ) {
        dim3 dimGrid(m);
        dim3 dimBlock(32);
        gemv_v1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n);
    }


    // for (int run = 0 ; run < iteration; run ++ ) {
    //     dim3 dimGrid(CEIL(m,4));
    //     dim3 dimBlock(4,32);
    //     gemv_v2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n);
    // }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    duration[0] = msecTotal / iteration;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

//




}