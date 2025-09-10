#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "parameters.h"
#include "helper.h"

void CUBLAS_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc);

void CUBLAS_MMult(cublasHandle_t handle, int m, int n, int k,
                       half* a, int lda,
                       half* b, int ldb,
                       half* c, int ldc);

void MY_MMult(cublasHandle_t handle, int m, int n, int k, half *d_A, int lda,
              half *d_B, int ldb, half *d_C, int ldc);

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc);

// 模板化测试函数
template <typename T>
void run_test(const char* type_name) {
    // 打印 GPU 信息
    cudaDeviceProp deviceProp;
    int devID = 0;
    checkCudaErrors(cudaSetDevice(devID));
    auto error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("[%s] GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           type_name, devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int p, m, n, k, rep;
    double diff;

    T *a, *b, *cold, *cref;

    printf("[%s] MY_MMult = [\n", type_name);

    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for (p = PFIRST; p <= PLAST; p += PINC) {
        m = (M == -1 ? p : M);
        n = (N == -1 ? p : N);
        k = (K == -1 ? p : K);

        const int lda = k, ldb = n, ldc = n;

        const size_t mem_size_A = m * k * sizeof(T);
        const size_t mem_size_B = k * n * sizeof(T);
        const size_t mem_size_C = m * n * sizeof(T);

        a    = (T*)malloc(mem_size_A);
        b    = (T*)malloc(mem_size_B);
        cold = (T*)malloc(mem_size_C);
        cref = (T*)malloc(mem_size_C);

        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        memset(cold, 0, mem_size_C);
        memset(cref, 0, mem_size_C);

        T *d_A, *d_B, *d_C;
        checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
        checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
        checkCudaErrors(cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, b, mem_size_B, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

        CUBLAS_MMult(handle, m, n, k, a, lda, b, ldb, cref, ldc);
        // REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);

        checkCudaErrors(cudaEventRecord(start, NULL));
        for (rep = 0; rep < NREPEATS; rep++) {
            MY_MMult(handle, m, n, k, d_A, k, d_B, n, d_C, n);
        }
        checkCudaErrors(cudaEventRecord(stop, NULL));
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        float msecPerMatrixMul = msecTotal / NREPEATS;
        double flopsPerMatrixMul = 2.0 * m * k * n;
        double gflops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

        checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        diff = compare_matrices(m, n, cold, ldc, cref, ldc);
        if (diff > 0.5f || diff < -0.5f) {
            printf("[%s] diff too big !\n", type_name);
        }
        printf("M=%d,N=%d,K=%d, gflops = %.2f, diff = %le \n", M, N, K, gflops, diff);

        free(a);
        free(b);
        free(cold);
        free(cref);

        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));
    }

    checkCudaErrors(cublasDestroy(handle));
    printf("[%s] ];\n", type_name);
}

// 主函数调用
int main() {
    // run_test<float>("float");
    run_test<half>("half");
    return 0;
}
