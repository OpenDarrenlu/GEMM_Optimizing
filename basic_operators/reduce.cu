#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"
#define WARP_SIZE 32
// #define CEIL(a,b) (a+b-1)/b
void reduce_1dim_cpu(float* input, float* output, int N){
    float sum = 0;
    for (int i = 0; i < N; i++){
        sum += input[i];
    }
    *output = sum;
}

void reduce_2dim_cpu(float* input, float* output, int M, int N){
    for(int i = 0 ; i < M ;i++){
        float sum = 0;
        for(int j = 0 ; j < N; j++){
            sum += input[i * N + j];
        }
        output[i] = sum;
    }
}
// dim3 block(256);
// dim3 grid(CEIL(N, 256));
//✅ 10.15
template<int BLOCK_SIZE>
__global__ void reduce_1dim(float* input, float* output, int N){
    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;
    const int warpNum = BLOCK_SIZE >> 5;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N)? input[tid] : 0.0f;

    __shared__ float s_y[32];

    //规约一个warp内
    for(int offset = warpSize >>1; offset > 0; offset >>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();

    //规约一个block内的warp
    if(warpId == 0){
        val = (laneId < warpNum)? s_y[laneId] : 0.0f;
        for(int offset = warpNum >> 1; offset > 0; offset >>=1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(laneId == 0) atomicAdd(output, val);
    }
}
void test_1dim(){
    const int N = 100000;
    float *h_input, *h_output,*h_output_reduce;
    float *d_input, *d_output;
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(sizeof(float));
    h_output[0] = 0.0f;
    h_output_reduce = (float*)malloc(sizeof(float));
  
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 100;
    }

    reduce_1dim_cpu(h_input, h_output, N);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(CEIL(N, BLOCK_SIZE));
    reduce_1dim<BLOCK_SIZE><<<grid, block>>>(d_input, d_output, N);
    cudaMemcpy(h_output_reduce, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output, h_output_reduce, 1);
    printf("%f\n", *h_output_reduce);
    printf("%f\n", *h_output);

    free(h_input);
    free(h_output);
    free(h_output_reduce);
    cudaFree(d_input);
    cudaFree(d_output);
}

// dim3 block(32);
// dim3 grid(M);
//✅ 10.15
template<int BLOCK_SIZE>
__global__ void reduce_2dim(float *input, float *output, int M, int N){
    const int lane_idx = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;
    // const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // const int partion = CEIL(N, blockDim.x );
    const int WarpNum = BLOCK_SIZE >> 5;
    if(blockIdx.x >= M) return ;
    __shared__ float sum_per_warp[WarpNum] ;
    // float sum_per_block = 0.0f;//错误的写法，这样每个thread都会执行output[blockIdx.x] = sum_per_block;但是只有thread0的是对的

    float val = 0.0f;

    for(int i = threadIdx.x; i < N ; i+=BLOCK_SIZE){
        int offset = i + blockIdx.x * N;
        val += input[offset];
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane_idx == 0) sum_per_warp[warpId] = val;
    __syncthreads();

    //warp内规约
    if(warpId == 0){
        val = (lane_idx < WarpNum)? sum_per_warp[lane_idx] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(lane_idx == 0) output[blockIdx.x] = val;
    }
}
// dim3 block(256);
// dim3 grid(M);
//✅ 10.15
template<int BLOCK_SIZE>
__global__ void reduce_2dim_v2(float *input, float *output, int M, int N){
    const int lane_idx = threadIdx.x % warpSize;
    const int warp_idx = threadIdx.x / warpSize;
    if(blockIdx.x >=M) return;
    const int partion = CEIL(N, blockDim.x);
    const int warpNum = CEIL(blockDim.x, warpSize);

    float val = 0.0f; //每个线程的寄存器的值
    for(int i = 0; i < partion ; i++){
        int col = i * blockDim.x + threadIdx.x;
        val +=(col < N)? (input[blockIdx.x * N + col]):0.0f;
    }
    
    //warp内规约
    __shared__ float s_y[32];
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane_idx == 0) s_y[warp_idx] = val;
    __syncthreads();

    //block内规约
    if(warp_idx == 0){
        val = (lane_idx < warpNum)? s_y[lane_idx] : 0.0f;
        for(int offset = warpNum >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(lane_idx == 0) output[blockIdx.x] = val;
    }
    
}
void test_2dim(){
    const int M = 2000;
    const int N = 1000; 
    float* h_input, *h_output;
    float* d_input, *d_output, *output;
    h_input = (float*)malloc(N * M* sizeof(float));
    h_output = (float*)malloc(M * sizeof(float));
    output = (float*)malloc(M * sizeof(float));
    for(int i = 0 ; i < N * M ; i++){
        h_input[i] = rand() % 1000;
    }

    reduce_2dim_cpu(h_input, h_output, M, N);

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * sizeof(float));
    cudaMemcpy(d_input, h_input, M * N* sizeof(float), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    reduce_2dim<BLOCK_SIZE><<<grid, block>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("%f\n", TIME_RECORD(100, ([&]{reduce_2dim<BLOCK_SIZE><<<grid, block>>>(d_input, d_output, M, N);})));


    verify_matrix(h_output, output, M);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(output);   
}
void test_2dim_v2(){
    const int M = 2000;
    const int N = 1000; 
    float* h_input, *h_output;
    float* d_input, *d_output, *output;
    h_input = (float*)malloc(N * M* sizeof(float));
    h_output = (float*)malloc(M * sizeof(float));
    output = (float*)malloc(M * sizeof(float));
    for(int i = 0 ; i < N * M ; i++){
        h_input[i] = rand() % 1000;
    }

    reduce_2dim_cpu(h_input, h_output, M, N);

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * sizeof(float));
    cudaMemcpy(d_input, h_input, M * N* sizeof(float), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    reduce_2dim_v2<BLOCK_SIZE><<<grid, block>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("%f\n", TIME_RECORD(100, ([&]{reduce_2dim_v2<BLOCK_SIZE><<<grid, block>>>(d_input, d_output, M, N);})));


    verify_matrix(h_output, output, M);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(output);   

}
int main(){
    printf("test_1dim\n");
    test_1dim();
    printf("test_2dim\n");
    test_2dim();
    printf("test_2dim_v2\n");
    test_2dim_v2();
    return 0;
}