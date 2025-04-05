#define __CUDA_BF16_TYPES_EXIST__
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <curand.h>
#include <algorithm>
#include <iomanip>
#include "rtopk/csrc/cuda/rtopk_kernel.cuh"
#include <torch/torch.h>

// 0: float, 1: __nv_bfloat16
#define DTYPE_IDX 1
#if DTYPE_IDX == 0
#define DTYPE float
#elif DTYPE_IDX == 1
#define DTYPE __nv_bfloat16
#endif

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(
    std::chrono::time_point<std::chrono::system_clock> a,
    std::chrono::time_point<std::chrono::system_clock> b) {
    return std::chrono::duration<double>(b - a).count();
}

using namespace std;

void checkErr() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

// kernel for converting float to __nv_bfloat16
__global__ void convert_float_to_bfloat16(float *input, __nv_bfloat16 *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

int main() {
    // int N_list[] = {16384, 65536, 262144, 1048576};
    int N_list[] = {8192};
    int dim_origin_list[] = {8192};
    // int dim_k_list[] = {16, 32, 64, 96, 128};
    int dim_k_list[] = {64, 128, 256, 512};
    int max_iter_list[] = {2, 3, 4, 5, 6, 7, 8, 10000};
    // int max_iter_list[] = {10000};
    float precision_list[] = {0};

    int max_N = *std::max_element(std::begin(N_list), std::end(N_list));
    int max_dim_origin = *std::max_element(std::begin(dim_origin_list),
                                           std::end(dim_origin_list));
    int max_dim_k =
        *std::max_element(std::begin(dim_k_list), std::end(dim_k_list));

    cout << "max N = " << max_N << ", preparing data..." << endl;

    DTYPE *value;
    int *index;

    cudaMallocManaged(&value, max_N * max_dim_k * sizeof(DTYPE));
    cudaMallocManaged(&index, max_N * max_dim_k * sizeof(int));

    curandGenerator_t gen;
    float *devDataFloat;
    cudaMalloc((void **)&devDataFloat, max_N * max_dim_origin * sizeof(float));

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, devDataFloat, max_N * max_dim_origin);
    #if DTYPE_IDX == 1
    DTYPE *devData;
    cudaMalloc((void **)&devData, max_N * max_dim_origin * sizeof(DTYPE));
    convert_float_to_bfloat16<<<(max_N + 255) / 256, 256>>>(
        devDataFloat, devData, max_N * max_dim_origin);
    #else
    float *devData = devDataFloat;
    #endif

    cudaDeviceSynchronize();
    checkErr();

    cout << "data ready, testing..." << endl;

    ofstream fout("output3000.txt");

    for (int N : N_list) {
        for (int dim_origin = 1024; dim_origin <= 8192; dim_origin *= 2) {
            // for (int dim_origin : dim_origin_list){
            for (int dim_k : dim_k_list) {
                if (dim_k >= dim_origin) {
                    continue;
                }
                for (float precision_float : precision_list) {
                    #if DTYPE_IDX == 1
                    DTYPE precision = __float2bfloat16(precision_float);
                    #else
                    DTYPE precision = (DTYPE)precision_float;
                    #endif
                    for (int max_iter : max_iter_list) {
                        cout << "N = " << N << ", dim_origin = " << dim_origin;

                        int w;
                        if (dim_origin <= 1024) {
                            w = 8;
                        } else if (dim_origin <= 2048) {
                            w = 4;
                        } else if (dim_origin <= 4096) {
                            w = 2;
                        } else {
                            w = 1;
                        }
                        int shared_size = w * dim_origin * sizeof(DTYPE);
                        int times = 4;
                        // warmup
                        for (int i = 0; i < times; i++) {
                            if (w == 8) {
                                rtopk_kernel<DTYPE, 8>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else if (w == 4) {
                                rtopk_kernel<DTYPE, 4>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else if (w == 2) {
                                rtopk_kernel<DTYPE, 2>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else {
                                rtopk_kernel<DTYPE, 1>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            }
                        }
                        cudaDeviceSynchronize();
                        double measured_time = 0;
                        for (int i = 0; i < times; i++) {
                            timestamp(t0);
                            if (w == 8) {
                                rtopk_kernel<DTYPE, 8>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else if (w == 4) {
                                rtopk_kernel<DTYPE, 4>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else if (w == 2) {
                                rtopk_kernel<DTYPE, 2>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            } else {
                                rtopk_kernel<DTYPE, 1>
                                    <<<N / w, w * 32, shared_size>>>(
                                        devData, value, index, N, dim_origin,
                                        dim_k, max_iter, precision);
                            }
                            cudaDeviceSynchronize();
                            checkErr();
                            timestamp(t1);
                            measured_time += getDuration(t0, t1);
                        }

                        cout
                             << ", dim_k = " << dim_k
                             << ", max_iter = " << max_iter
                             << ", rtopk time = " << measured_time / times * 1000 << " ms" << endl;
                        fout << "N = " << N << ", dim_origin = " << dim_origin
                             << ", dim_k = " << dim_k
                             << ", max_iter = " << max_iter
                             << ", rtopk time = " << measured_time / times * 1000 << " ms" << endl;
                        // fout.flush();

                        // // Convert raw CUDA data to torch tensors for comparison
                        // // Create a torch tensor that views the raw CUDA data without copying
                        // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
                        // torch::Tensor data_tensor = torch::from_blob(devData, {N, dim_origin}, options);

                        // // Create output tensors
                        // torch::Tensor values_tensor = torch::from_blob(value, {N, dim_k}, options);
                        // torch::Tensor indices_tensor = torch::from_blob(index, {N, dim_k}, 
                        //                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

                        // double measured_time_topk = 0;
                        // cudaDeviceSynchronize();
                        // for (int i = 0; i < times; i++) {
                        //     timestamp(t0);
                            
                        //     auto result = torch::topk(data_tensor, dim_k, 1);

                        //     cudaDeviceSynchronize();
                        //     checkErr();
                        //     timestamp(t1);
                        //     measured_time_topk += getDuration(t0, t1);
                        // }

                        // cout << ", torch topk time = " << measured_time_topk / times * 1000 << " ms" << endl;
                        // fout << ", torch topk time = " << measured_time_topk / times * 1000 << " ms" << endl;
                    }
                }
            }
        }
    }

    fout.close();

    // Free unified memory
    cudaFree(value);
    cudaFree(index);
    curandDestroyGenerator(gen);
    cudaFree(devData);

    cout << "finish" << endl;

    return 0;
}