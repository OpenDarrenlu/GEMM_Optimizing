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
//未写完。 懂大概就行，频率不高
void batchnorm_2dim_cpu(float *input, float *output, int N, int C, float *mean, float *var, float *gamma, float *beta, float eps){
    //[N,C]
    for(int c = 0 ;c < C; c++){
        //  step1: 计算当前通道c的均值
        float mean =0.0f;
        for(int n = 0 ; n < N; n++){
            mean += input[n * C +c];
        }
        mean /= N;
        
        //  step2: 计算当前通道c的方差
        float var = 0.0f;
        for(int n = 0 ; n < N; n++){
            var += (input[n * C +c] - mean) * (input[n * C +c] - mean);
        }
        var /= N;   

        //step3: 标准化
        for(int n = 0 ; n < N; n++){
            float x = input[n * C +c];
            output[n * C +c] = gamma[c] * (x - mean) / sqrt(var + eps) + beta[c];
        }
    }
}
void batchnorm_2dim_cpu_2(float *input, float *output, int N, int C, float *mean, float *var, float *gamma, float *beta, float eps){
    //[N,C]
    float mean[C] = {0};
    float var[C] = {0};
    for(int i = 0 ; i < N; i ++){
        for(int c = 0; c <C;c++){
            mean[c] += input[i * C + c];
        }
    }

}
// void batchnorm_4dim_cpu(float *input, float *output, int N, int C, int H, int W, float *mean, float *var, float *gamma, float *beta, float eps){
//     //[N,C,H,W]
//     for(int c = 0 ;c < C; c++){


//     }
// }
// void test_batchnorm_4dim(){
//     //[N,C,H,W]
//     //对每个通道 C，独立地计算该通道在所有 N 个样本、所有 H × W 空间位置上的均值和方差，
//     //然后对该通道的所有 N×H×W 个值进行标准化


// }

void test_batchnorm_2dim(){
    //[N,C] N为batchsize(样本数),C为通道数
    const int N = 64;  
    const int C = 128;
    float *h_input, *h_output, *h_output_ref, *h_gamma, *h_beta;
    float *d_input, *d_output;
    batchnorm_2dim_cpu(h_input, h_output, N, C, h_gamma, h_beta, 0.00001);
    batchnorm_2dim_cpu_2(h_input, h_output_ref, N, C, h_gamma, h_beta, 0.00001);
    verify_matrix(h_output, h_output_ref, N, C);
}
int main(){
    test_batchnorm_2dim();
    // test_batchnorm_4dim();
}