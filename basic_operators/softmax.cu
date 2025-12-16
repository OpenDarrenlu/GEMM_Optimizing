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

__device__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_int = (int*)address;  // float 的内存表示与 int 一样都是 4 字节
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        // 把当前 float 值的位模式转成 int，再转回 float 做比较
        float current = __int_as_float(old);
        if (current >= val) break;  // 如果当前值已经 >= val，就不更新
        // 把 val 的位模式转成 int
        old = atomicCAS(address_as_int, old, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(*address_as_int);
}
void softmax_1dim_cpu(float *input, float *output, int N){
    //max
    float max = -FLT_MAX;
    for(int i = 0 ; i < N ; i++){
        max = std::max(max, input[i]);
    }
    printf("max: %f\n", max);
    //exp_sum
    float exp_sum = 0.0f;
    for(int i = 0 ; i < N ; i++){
        exp_sum += exp(input[i] - max);
    }
    printf("exp_sum: %f\n", exp_sum);

    float exp_sum_inverse = 0.0f;
    for(int i = N-1; i >= 0 ; i--){
        exp_sum_inverse += exp(input[i] - max);
    }
    printf("exp_sum_inverse: %f\n", exp_sum_inverse);
    //softmax
    for(int i = 0 ; i < N ; i++){
        output[i] = exp(input[i] - max) / exp_sum;
    }   
}

void softmax_2dim_cpu(float *input, float *output, int M, int N){
    for(int i = 0 ; i < M ; i++){
        //max
        float max = -FLT_MAX;
        for(int j = 0; j < N ;j++){
            max = std::max(max, input[i*N + j]);
        }
         //sum_epf
        float exp_sum = 0.0f;
        for(int j = 0; j < N ;j++){
            exp_sum += exp(input[i*N + j] - max);
        }
        //softmax
        if(i == 0){
            // printf("max: %f\n", max);
            // printf("exp_sum: %f\n", exp_sum);
        }
        for(int j = 0; j < N ;j++){
            output[i*N + j] = exp(input[i*N + j] - max) / exp_sum;
        }
    }
}
void softmax_2dim_cpu_online(float *input, float *output, int M, int N){
    //max[i]= max(max[i-1], xi)
    //exp_sum[i] = exp_sum[i-1] * expf(max[i-1]- max[i])+ exp(xi - max[i])
    for(int i = 0 ; i < M ; i++){
        float old_max = -FLT_MAX;
        float new_max = -FLT_MAX;
        float exp_sum = 0.0f;
        for(int j = 0; j < N ;j++){
            new_max = std::max(new_max, input[i*N + j]);
            exp_sum = exp_sum*exp(old_max - new_max) +exp(input[i*N + j] - new_max);
            old_max = new_max;
        }
        if(i == 0){
            // printf("max: %f\n", new_max);
            // printf("exp_sum: %f\n", exp_sum);
        }
        for(int j = 0; j < N ;j++){
            output[i*N + j] = exp(input[i*N + j] - new_max) / exp_sum;
        }
    }
}
// dim3 block(256);
// dim3 grid(CEIL(N, 256));
__global__ void max_kernel(float *input, int N ,float *max){
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_num = blockDim.x / warpSize;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= N) return;
    // float max_val = -FLT_MAX;
    float val = input[tid];
    __shared__ float s_max[32];


    // 规约一个warp
    for(int offset = warpSize ; offset > 0; offset >>=1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if(lane_id == 0) s_max[warp_id] = val;
    __syncthreads();


    //规约一个block
    if(warp_id==0){
        val = (lane_id < warp_num) ? s_max[lane_id] : -FLT_MAX;
        for(int offset = warpSize ; offset > 0; offset >>=1){
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if(lane_id == 0) atomicMaxFloat(max, val);
    }
}
// dim3 block(256);
// dim3 grid(CEIL(N, 256));
__global__ void exp_sum_kernel(float *input, int N ,float *max, float *exp_sum){
    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;
    const int warpNum = blockDim.x / warpSize;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N)? exp(input[tid] - *max) : 0.0f;

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
        if(laneId == 0) atomicAdd(exp_sum, val);
    }
}
__global__ void softmax_kernel(float *input, int N ,float *max, float *exp_sum, float *output){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= N) return;
    float val = exp(input[tid] - *max) / *exp_sum;
    output[tid] = val;
}



// dim3 block(32);
// dim3 grid(M);
__global__ void softmax_2dim(float *input,float *output,int M ,int N){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int partion = CEIL(N, blockDim.x);
    // const int warp_idx = threadIdx.x / warpSize;
    const int lane_idx = threadIdx.x % warpSize;

    //max
    float max_val = -FLT_MAX;
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        if(col < N){
            max_val = fmax(max_val, input[blockIdx.x * N + col]);
        }
    }

    //规约到一个warp内
    __shared__  float smm_max;
    for(int offset = warpSize >> 1; offset > 0; offset >>=1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if(lane_idx == 0) smm_max= max_val;
    __syncthreads();


    //sum_exp
    float sum_exp = 0.0f;
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        sum_exp += (col < N) ? exp(input[col + blockIdx.x * N] - smm_max) : 0.0f;
    }
    //规约到一个warp内
    __shared__  float sum_epf;
    for(int offset = warpSize >> 1; offset > 0; offset >>=1){
        sum_exp +=  __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    //online-2pass
    //max[i] = max(max[i-1], x[i]);
    //sum_epf[i] = sum_epf[i-1]* epf(max[i-1]- max[i]) + epf(x[i]- max[i]);

    if(lane_idx == 0) sum_epf= sum_exp;
    __syncthreads();    

    //softmax
    // for(int i = 0 ; i < N ;i++){
    //     output[blockIdx.x * N + i] = expf(input[blockIdx.x * N + i] - smm_max) / sum_epf;
    // }
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        output[blockIdx.x * N+ col] = (col < N) ? exp(input[blockIdx.x * N+ col] - smm_max) / sum_epf : 0.0f;
    }
}



// dim3 block(256);
// dim3 grid(M);
__global__ void softmax_2dim_v2(float *input,float *output,int M ,int N){
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int partion = CEIL(N, blockDim.x);
    const int warp_num = blockDim.x / warpSize;

    //max
    float max_val = -FLT_MAX;
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        if(col < N){
            max_val = max(max_val, input[col + blockIdx.x * N]);
        }
    }
    //规约到一个warp内
    __shared__  float sm_max[32];
    for(int offset = warpSize >>1; offset >0 ; offset >>=1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if(lane_id == 0) sm_max[warp_id] = max_val;
    __syncthreads();

    //规约到一个block内
    if(warp_id == 0){
        max_val = (lane_id < warp_num) ? sm_max[lane_id] : -FLT_MAX;
        for(int offset = warpSize >>1; offset >0 ; offset >>=1){
            max_val = fmax(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if(lane_id == 0) sm_max[0] = max_val; 
    }
    __syncthreads();

    //sum_exp
    float sum_exp = 0.0f;
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        sum_exp += (col < N) ? exp(input[col + blockIdx.x * N] - sm_max[0]) : 0.0f;
    }
    //规约到一个warp内
    __shared__ float sm_sum_exp[32];
    for(int offset = warpSize >>1; offset >0 ; offset >>=1){
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    if(lane_id==0) sm_sum_exp[warp_id] = sum_exp;
    __syncthreads();

    //将多个warp规约到block内
    if(warp_id==0){
        sum_exp = (lane_id < warp_num) ? sm_sum_exp[lane_id] : 0.0f;
        for(int offset = warpSize >>1; offset >0 ;offset >>=1){
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        if(lane_id==0) sm_sum_exp[0] = sum_exp;
    }
    __syncthreads();
    
    //softmax
    // for(int i = 0 ; i < N; i++){
    //     output[blockIdx.x * N + i] = exp(input[blockIdx.x * N + i] - sm_max[0]) / sm_sum_exp[0];
    // }
    for(int i = 0; i < partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        if(col < N){
            output[blockIdx.x * N + col] = exp(input[blockIdx.x * N + col] - sm_max[0]) / sm_sum_exp[0];
        }
    }
}

void test_1dim_softmax(){
    int N = 1024*1024 ;
    float *h_input , *h_output;
    float *d_input, *d_output;

    h_input = (float *)malloc(N * sizeof(float));
    h_output = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    softmax_1dim_cpu(h_input, h_output, N);

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    //max_kernel
    dim3 block(256);
    dim3 grid(CEIL(N, 256));
    float *d_max;
    cudaMalloc(&d_max, sizeof(float));
    max_kernel<<<grid, block>>>(d_input, N,d_max);

    float *max = (float *)malloc(sizeof(float));
    cudaMemcpy(max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    printf("max: %f\n", *max);
 
    // //exp_sum_kernel
    float *d_exp_sum;
    cudaMalloc(&d_exp_sum, sizeof(float));
    exp_sum_kernel<<<grid, block>>>(d_input, N,d_max,d_exp_sum);    
    float *exp_sum = (float *)malloc(sizeof(float));
    cudaMemcpy(exp_sum, d_exp_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("exp_sum: %f\n", *exp_sum);



    //softmax_kernel
    softmax_kernel<<<grid, block>>>(d_input, N,d_max,d_exp_sum,d_output);
    float *h_output_gpu = (float *)malloc(N * sizeof(float));
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    verify_matrix(h_output, h_output_gpu, N);
    




}

void test_2dim_softmax(){
    int M = 1024;
    int N = 1024;
    float *h_input, *h_output, *h_output_gpu, *h_output_online;
    float *d_input, *d_output;  
    
    h_input = (float *)malloc(M * N * sizeof(float));
    h_output = (float *)malloc(M * N * sizeof(float));
    h_output_online = (float *)malloc(M * N * sizeof(float));
    h_output_gpu = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++){
        h_input[i] = rand() / (float)RAND_MAX;
    }
    // printf("softmax_2dim_cpu\n");
    softmax_2dim_cpu(h_input, h_output, M, N);
    // printf("softmax_2dim_cpu_online\n");
    softmax_2dim_cpu_online(h_input, h_output_online, M, N);
    // printf("verify_matrix_cpu\n");
    // verify_matrix(h_output, h_output_online, M * N);
    // printf("===============================\n");

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(32);
    dim3 grid(M);
    softmax_2dim<<<grid, block>>>(d_input, d_output, M, N);
    cudaMemcpy(h_output_gpu, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_matrix(h_output, h_output_gpu, M * N);

}

void test_2dim_softmax_v2(){
    int M = 1024;
    int N = 1024;
    float *h_input, *h_output, *h_output_gpu, *h_output_online;
    float *d_input, *d_output;  
    
    h_input = (float *)malloc(M * N * sizeof(float));
    h_output = (float *)malloc(M * N * sizeof(float));
    h_output_online = (float *)malloc(M * N * sizeof(float));
    h_output_gpu = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++){
        h_input[i] = rand() / (float)RAND_MAX;
    }
    printf("softmax_2dim_cpu\n");
    softmax_2dim_cpu(h_input, h_output, M, N);
    printf("softmax_2dim_cpu_online\n");
    softmax_2dim_cpu_online(h_input, h_output_online, M, N);
    printf("verify_matrix_cpu\n");
    verify_matrix(h_output, h_output_online, M * N);
    // printf("===============================\n");

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(256);
    dim3 grid(M);
    softmax_2dim_v2<<<grid, block>>>(d_input, d_output, M, N);
    cudaMemcpy(h_output_gpu, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_matrix(h_output, h_output_gpu, M * N);

}

int main(){
    test_1dim_softmax();
    printf("test_2dim_softmax\n");
    test_2dim_softmax();
    printf("test_2dim_softmax_v2\n");
    test_2dim_softmax_v2();
}