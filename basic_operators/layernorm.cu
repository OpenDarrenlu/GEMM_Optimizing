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

const float eps = 1e-5f;  
void layer_norm_2dim_cpu(float *input, float *output, int seq_len, int hidden_size, float *gamma, float *beta) {
    const float eps = 1e-5f;  // 防止除以零的小常数

    for (int i = 0; i < seq_len; i++) {  // 遍历每个 token（每一行）
        float mean = 0.0f;
        float var = 0.0f;

        // 1. 计算均值 mean
        for (int j = 0; j < hidden_size; j++) {
            float x = input[i * hidden_size + j];
            mean += x;
        }
        mean /= hidden_size;
        // printf("mean = %f\n", mean);

        // 2. 计算方差 var = E[(x - mean)^2]
        for (int j = 0; j < hidden_size; j++) {
            float x = input[i * hidden_size + j];
            var += (x - mean) * (x - mean);
        }
        var /= hidden_size;
        // printf("var = %f\n", var);

        // 3. 标准化 + 缩放偏移（仿射变换）
        for (int j = 0; j < hidden_size; j++) {
            float x = input[i * hidden_size + j];
            float normalized = (x - mean) / sqrtf(var + eps);  // 标准化
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];  // 仿射变换
            // printf("output[%d] = %f\n", i * hidden_size + j, output[i * hidden_size + j]);
        }
    }
}


void layer_norm_3dim_cpu(float *input, float *output, int batch_size, int seq_len, int hidden_size, float *gamma, float *beta){
    const float eps = 1e-5f;  
    for(int i = 0 ; i < batch_size ;i ++){
        for(int j = 0 ; j < seq_len ; j++){
            // 计算当前 token 在输入中的起始索引
            int start_index = i * seq_len * hidden_size + j * hidden_size;
            //求mean
            float mean = 0.0f;
            for(int k = 0 ; k < hidden_size ; k++){
                mean += input[start_index + k];
            }
            mean /= hidden_size;
            //求var
            float var = 0.0f;
            for(int k = 0 ; k < hidden_size ; k++){
                var += (input[start_index + k] - mean) * (input[start_index + k] - mean);
            }
            var /= hidden_size;
            //标准化
            for(int k = 0 ; k < hidden_size ; k++){
                output[start_index + k] = (input[start_index + k] - mean) / sqrtf(var + eps) * gamma[k] + beta[k];
                if(i == 0 && j == seq_len-1 && k == hidden_size-1){
                    // printf("output[%d] = %f\n", start_index + k, output[start_index + k]);
                }
            }   
        }
    
    }
}

//dim3 block(256);
//dim3 gird(seq_len, batch_size);
__global__ void layernorm_3dim(float *input, float *output, int batch_size, int seq_len, int hidden_size, float *gamma, float *beta){
    // const index = i * seq_len * hidden_size + j * hidden_size;
    // const int start_index = blockIdx.x * seq_len * hidden_size + blockIdx.y * hidden_size;
    const int start_index = blockIdx.y * seq_len * hidden_size + blockIdx.x * hidden_size;
    const int warp_id =  threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_num = blockDim.x / warpSize;
    const int partion = CEIL(hidden_size, blockDim.x);

    // 1. 计算均值 mean
    float val = 0.0f;
    for(int i = 0 ; i < partion; i++ ){
        int col  = i * blockDim.x + threadIdx.x;
        val += (col < hidden_size) ? input[start_index + col] : 0.0f;
    }
    //规约一个warp
    __shared__ float sm_sum[32];
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane_id == 0) sm_sum[warp_id] = val;
    __syncthreads();

     //规约所有warp
    if(warp_id == 0){
        val = (lane_id < warp_num)? sm_sum[lane_id] : 0.0f;
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(lane_id==0) sm_sum[0]= val/hidden_size;
    }
    __syncthreads();    

    //2. 计算方差 var = E[(x - mean)^2]
    float var = 0.0f;
    for(int i = 0 ; i < partion; i++ ){
        int col  = i * blockDim.x + threadIdx.x;
        var += (col < hidden_size) ? (input[start_index + col] - sm_sum[0]) * (input[start_index + col] - sm_sum[0]) : 0.0f;
    }
    //规约一个warp
    __shared__ float sm_var[32];
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    if(lane_id == 0) sm_var[warp_id] = var;
    __syncthreads();        

    //规约所有warp
    if(warp_id == 0){
        var = (lane_id < warp_num)? sm_var[lane_id] : 0.0f;
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
        if(lane_id==0) sm_var[0]= var/hidden_size;
    }
    __syncthreads();


    //3. 标准化 + 缩放偏移（仿射变换）
    // for(int i = 0 ; i < hidden_size ; i++ ){
    //     output[start_index + i] = (input[start_index + i] - sm_sum[0]) / sqrtf(sm_var[0] + eps) * gamma[i] + beta[i];
    // }
    for(int i = 0 ; i <  partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        output[start_index + col] = (input[start_index + col] - sm_sum[0]) / sqrtf(sm_var[0] + eps) * gamma[col] + beta[col];
    }

}
// dim3 block(256);
// dim3 grid(seq_len);
__global__ void layernorm_2dim(float *input, float *output, int seq_len, int hidden_size, float *gamma, float *beta) {
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_num = blockDim.x / warpSize;
    const int partion = CEIL(hidden_size, blockDim.x);

    // 1. 计算均值 mean
    float val = 0.0f;
    for(int i = 0 ; i <  partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        val += (col < hidden_size) ?  input[blockIdx.x * hidden_size + col] : 0.0f;
    }
    //规约一个warp
    __shared__ float sm_sum[32];
    for(int offset =warpSize / 2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane_id == 0) sm_sum[warp_id] = val;
    __syncthreads();

    //规约所有warp
    if(warp_id == 0){
        val = (lane_id < warp_num)? sm_sum[lane_id] : 0.0f;
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(lane_id==0) sm_sum[0]= val/hidden_size;
    }
    __syncthreads();

    //2. 计算方差 var = E[(x - mean)^2]
    float var = 0.0f;
    for(int i = 0 ; i <  partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        var += (col < hidden_size) ?  (input[col + blockIdx.x * hidden_size] - sm_sum[0]) * (input[col + blockIdx.x * hidden_size] - sm_sum[0]) : 0.0f;
    }
    
    //规约一个warp
    __shared__ float sm_var[32];
    for(int offset =warpSize / 2; offset > 0; offset /= 2){
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    if(lane_id == 0) sm_var[warp_id] = var;
    __syncthreads();

    if(warp_id == 0){
        var = (lane_id < warp_num)? sm_var[lane_id] : 0.0f;
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
        if(lane_id==0) sm_var[0]= var/hidden_size;
    }
    __syncthreads();

    // if (blockIdx.x == 0 && warp_id == 0 && lane_id == 0) {  // 只让第一个 token 打印
    //     float mean =  sm_sum[0];
    //     float variance = sm_var[0];
    //     printf("[DEBUG]  Mean = %f, Variance = %f\n", mean, variance);
    // }

    //3. 标准化 + 缩放偏移（仿射变换）
    // for(int i = 0 ;i < hidden_size ;i ++){
    //     output[i + blockIdx.x * hidden_size] = (input[i + blockIdx.x * hidden_size] - sm_sum[0]) / sqrtf(sm_var[0] + eps) * gamma[i] + beta[i];
    // }
    for(int i = 0 ; i <  partion; i++){
        int col = i * blockDim.x + threadIdx.x;
        output[blockIdx.x * hidden_size + col] = (input[blockIdx.x * hidden_size + col] - sm_sum[0]) / sqrtf(sm_var[0] + eps) * gamma[col] + beta[col];
    }
}


void test_layer_norm_2dim(){
    const int seq_len = 1024;
    const int hidden_size = 1024;

    float *h_input, *h_output, *h_gamma, *h_beta, *h_output_gpu;
    float *d_input, *d_output, *d_gamma, *d_beta;
    
    h_input = (float *)malloc(seq_len * hidden_size * sizeof(float));
    h_output = (float *)malloc(seq_len * hidden_size * sizeof(float));
    h_output_gpu = (float *)malloc(seq_len * hidden_size * sizeof(float));
    h_gamma = (float *)malloc(hidden_size * sizeof(float));
    h_beta = (float *)malloc(hidden_size * sizeof(float));
    for(int i = 0; i < seq_len * hidden_size; i++){
        h_input[i] = (float)rand() / RAND_MAX;
    }
    for(int i = 0; i < hidden_size; i++){
        h_gamma[i] = (float)rand() / RAND_MAX;
        h_beta[i] = (float)rand() / RAND_MAX;
    }
    layer_norm_2dim_cpu(h_input, h_output, seq_len, hidden_size, h_gamma, h_beta);


    cudaMalloc(&d_input, seq_len * hidden_size * sizeof(float));
    cudaMalloc(&d_output, seq_len * hidden_size * sizeof(float));
    cudaMalloc(&d_gamma, hidden_size * sizeof(float));
    cudaMalloc(&d_beta, hidden_size * sizeof(float));
    cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(256);
    dim3 grid(seq_len);
    
    layernorm_2dim<<<grid, block>>>(d_input, d_output, seq_len, hidden_size, d_gamma, d_beta);
    cudaDeviceSynchronize();  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(h_output_gpu, d_output, seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", TIME_RECORD(100, ([&]{layernorm_2dim<<<grid, block>>>(d_input, d_output, seq_len, hidden_size, d_gamma, d_beta);})));


    verify_matrix(h_output, h_output_gpu, seq_len*hidden_size);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_output_gpu);
    free(h_gamma);
    free(h_beta);   

}
void test_layer_norm_3dim(){

    const int batch_size = 12;
    const int seq_len = 1024;
    const int hidden_size = 2048;

    
    float *h_input, *h_output, *h_gamma, *h_beta, *h_output_gpu;
    float *d_input, *d_output, *d_gamma, *d_beta;
    
    h_input = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    h_output = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    h_output_gpu = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    h_gamma = (float *)malloc(hidden_size * sizeof(float));
    h_beta = (float *)malloc(hidden_size * sizeof(float));
    for(int i = 0; i < batch_size * seq_len * hidden_size; i++){
        h_input[i] = (float)rand() / RAND_MAX;
    }   
    for(int i = 0; i < hidden_size; i++){
        h_gamma[i] = (float)rand() / RAND_MAX;
        h_beta[i] = (float)rand() / RAND_MAX;
    }
    layer_norm_3dim_cpu(h_input, h_output, batch_size,seq_len, hidden_size, h_gamma, h_beta);
    
    cudaMalloc(&d_input, batch_size * seq_len * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * hidden_size * sizeof(float));
    cudaMalloc(&d_gamma, hidden_size * sizeof(float));
    cudaMalloc(&d_beta, hidden_size * sizeof(float));
    cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    dim3  block(256);
    dim3  grid(seq_len, batch_size);
    layernorm_3dim<<<grid, block>>>(d_input, d_output, batch_size, seq_len, hidden_size, d_gamma, d_beta);
    cudaDeviceSynchronize();  
    printf("%f\n", TIME_RECORD(100, ([&]{layernorm_3dim<<<grid, block>>>(d_input, d_output, batch_size, seq_len, hidden_size, d_gamma, d_beta);})));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_output_gpu, d_output, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    verify_matrix(h_output, h_output_gpu, batch_size * seq_len * hidden_size);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_output_gpu);
    free(h_gamma);
    free(h_beta);       


}
int main(){
    printf("test_layer_norm_2dim\n");
    test_layer_norm_2dim();
    printf("======================\n");
    printf("test_layer_norm_3dim\n");
    test_layer_norm_3dim();
}