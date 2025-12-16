#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>
// #include <alorigthm>
#include <algorithm>
#include "utils.cuh"
// 定义INFINITY常量，如果编译器不支持，可以使用宏定义
#ifndef INFINITY
#define INFINITY std::numeric_limits<float>::infinity()
#endif

//h_a一行一行读, h_b一列一列写
void host_transpose1(float *h_a, float *h_b, const int M, const int N){
    //M * N --> N * M
    for(int i =0; i<M ;i++){
        for(int j = 0 ; j<N ;j++){
            h_b[j * M + i] = h_a[i * N + j]; //h_a一行一行读, h_b一列一列写
        }
    }
}
////h_a 一列一列读，h_b一行一行写
void host_transpose2(float *h_a, float *h_b, const int M, const int N){
    //M * N --> N * M
    for(int i =0; i< N ;i++){
        for(int j = 0; j< M ;j++){
            h_b[i * M + j] =  h_a[j * N + i];
        }
    }
}
//朴素实现，// 根据input的形状(M行N列)进行切块
//device_transpose_v0：读操作是合并的，写操作是不合并的
// grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M , BLOCK_SIZE));
__global__ void device_transpose_v0(float *d_input, float *d_output0, const int M , const int N){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col < N){
        d_output0[row + col * M] = d_input[col + row * N];
    }
}

//grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N , BLOCK_SIZE));
//device_transpose_v1：读操作是不合并的，写操作是合并的，速度提升
__global__ void device_transpose_v1(float *d_input, float *d_output1, const int M , const int N){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < N && col < M){
        d_output1[row * M + col]  = d_input[col * N + row];
    }
}

//device_transpose_v2：利用共享内存中转，读操作和写操作都是合并的，但是存在 bank conflict
//dim2 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  
template<const int BLOCK_SIZE>
__global__ void device_transpose_v2(float *d_input, float *d_output2, const int M, const int N){
    __shared__ float s[BLOCK_SIZE][BLOCK_SIZE];
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col <N){
        s[threadIdx.y][threadIdx.x] = d_input[row * N + col];
    }
    __syncthreads();
    const int dst_row = blockDim.x * blockIdx.x + threadIdx.y;
    const int dst_col = blockDim.y * blockIdx.y + threadIdx.x;
    if(dst_row < N && dst_col < M){
        // 合并写入，但是存在bank冲突：
        // 可以看出，同一个warp中的32个线程（连续的32个threaIdx.x值）
        // 将对应共享内存中跨度为32的数据，也就说，这32个线程恰好访问
        // 同一个bank中的32个数据，这将导致32路bank冲突
       d_output2[dst_row * M + dst_col] = s[threadIdx.x][threadIdx.y];
    }
}
// 使用共享内存中转，合并读取+写入，对共享内存做padding，解决bank conflict
//dim2 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  
template<const int BLOCK_SIZE>
__global__ void device_transpose_v3(float *d_input, float *d_output3, const int M, const int N){
    __shared__ float s[BLOCK_SIZE][BLOCK_SIZE+1]; // // 对共享内存做padding，解决bank conflict
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col <N){
        s[threadIdx.y][threadIdx.x] = d_input[row * N + col];
    }
    __syncthreads();
    const int dst_row = blockDim.x * blockIdx.x + threadIdx.y;
    const int dst_col = blockDim.y * blockIdx.y + threadIdx.x;
    if(dst_row < N && dst_col < M){
        // 通过做padding后，同一个warp中的32个线程（连续的32个threaIdx.x值）
        // 将对应共享内存中跨度为33的数据
        // 如果第一个线程访问第一个bank中的第一层
        // 那么第二个线程访问第二个bank中的第二层
        // 以此类推，32个线程访问32个不同bank，不存在bank冲突
       d_output3[dst_row * M + dst_col] = s[threadIdx.x][threadIdx.y];
    }
}
// 使用共享内存中转，合并读取+写入，使用swizzling解决bank conflict
template<const int BLOCK_SIZE>
__global__ void device_transpose_v4(float *d_input, float *d_output4, const int M, const int N){
    __shared__ float s[BLOCK_SIZE][BLOCK_SIZE];
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col <N){
        s[threadIdx.y][threadIdx.x ^ threadIdx.y] = d_input[row * N + col];
    }
    __syncthreads();
    const int dst_row = blockDim.x * blockIdx.x + threadIdx.y;
    const int dst_col = blockDim.y * blockIdx.y + threadIdx.x;
    if(dst_row <  N && dst_col <M){
        // swizzling主要利用了异或运算的以下两个性质来规避bank conflict：
        d_output4[dst_row * M + dst_col] = s[threadIdx.x][threadIdx.y^ threadIdx.x];
    }
}



int main(){
    const int M = 12800;
    const int N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 50;

    float *h_a, *h_b, *d_input;
    h_a = (float*)malloc(sizeof(float) * M * N);
    h_b = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i<M; i++){
        for(int j = 0; j<N ;j++){
            h_a[i * N + j] = i + j;
        }
    }
    for(int i = 0 ; i <N ;i++){
        for(int j = 0; j<M ;j++){
            h_b[i * M + j] = 0;
        }
    }
    cudaMalloc(&d_input, sizeof(float) * M * N);
    cudaMemcpy(d_input, h_a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    //host
    printf("begin to cal host\n");
    host_transpose2(h_a, h_b, M, N);
    // free(h_b);
    

    //device1: 朴素实现
    printf("begin to device1\n");
    float *d_output0;
    cudaMalloc(&d_output0, sizeof(float) * N * M);
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);

    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid_size0(CEIL(M, BLOCK_SIZE), CEIL(N , BLOCK_SIZE));
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M , BLOCK_SIZE)); //x维度基于N(列数)，y维度基于M(行数)
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_input, d_output0, M, N);}));
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool ver0 = verify_matrix(h_output0, h_b, M * N);    
    printf("ver0 = %d\n", ver0);
    printf("[device_transpose_v0] Average time: (%f) ms\n", total_time0 / repeat_times);  // 输出平均耗时

    cudaFree(d_output0);
    free(h_output0);

    //device1:
    float *d_output1;
    cudaMalloc(&d_output1, sizeof(float) * N * M);                             
    float *h_output1 = (float *)malloc(sizeof(float) * N * M);                         

    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(M,BLOCK_SIZE), CEIL(N, BLOCK_SIZE));
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_input, d_output1, M, N);}));
    cudaMemcpy(h_output1, d_output1, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool ver1 = verify_matrix(h_output1, h_b, M * N);    
    printf("ver1 = %d\n", ver1);
    printf("[device_transpose_v1] Average time: (%f) ms\n", total_time1 / repeat_times);  // 输出平均耗时

    cudaFree(d_output1);
    free(h_output1);

    //device2:    float *d_output2;
    float *d_output2;
    cudaMalloc(&d_output2, sizeof(float) * N * M);                             
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);                         

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<BLOCK_SIZE><<<grid_size2, block_size2>>>(d_input, d_output2, M, N);}));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool ver2 = verify_matrix(h_output2, h_b, M * N);    
    printf("ver2 = %d\n", ver2);
    printf("[device_transpose_v2] Average time: (%f) ms\n", total_time2 / repeat_times);  // 输出平均耗时

    cudaFree(d_output2);
    free(h_output2);

    //device3
    float *d_output3;
    cudaMalloc(&d_output3, sizeof(float) * N * M);                             
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);                         

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_input, d_output3, M, N);}));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool ver3 = verify_matrix(h_output3, h_b, M * N);    
    printf("ver3 = %d\n", ver3);
    printf("[device_transpose_v3] Average time: (%f) ms\n", total_time3 / repeat_times);  // 输出平均耗时

    cudaFree(d_output3);
    free(h_output3);

    //device4
    float *d_output4;
    cudaMalloc(&d_output4, sizeof(float) * N * M);                             
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);                         

    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  
    float total_time4 = TIME_RECORD(repeat_times, ([&]{device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_input, d_output4, M, N);}));
    cudaMemcpy(h_output4, d_output4, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool ver4 = verify_matrix(h_output4, h_b, M * N);    
    printf("ver4 = %d\n", ver4);
    printf("[device_transpose_v4] Average time: (%f) ms\n", total_time4 / repeat_times);  // 输出平均耗时

    cudaFree(d_output4);
    free(h_output4);





}

