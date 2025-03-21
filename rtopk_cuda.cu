#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "rtopk_kernel.cuh"  // Make sure this file is in your include path

// Convenience macros for error checking
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define RTOPK_CALL(DTYPE, CAST_DTYPE, W) \
    rtopk_kernel<DTYPE, W><<<blocks, threads, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>( \
        (DTYPE*)data.data_ptr<CAST_DTYPE>(), \
        (DTYPE*)values.data_ptr<CAST_DTYPE>(), \
        indices.data_ptr<int>(), \
        N, \
        dim_origin, \
        k, \
        max_iter, \
        precision_converted \
    )

// Wrapper function that launches the CUDA kernel
// Expects a 2D tensor 'data' of shape [N, dim_origin] and returns a tuple (values, indices),
// where for each of the N rows, the top-k approximate values (and their original indices)
// are stored in an output tensor of shape [N, k].
std::tuple<torch::Tensor, torch::Tensor> rtopk_forward_cuda(
    torch::Tensor data,
    int k,
    int max_iter,
    float precision) 
{
    // Ensure input is a contiguous CUDA tensor.
    CHECK_INPUT(data);
    AT_ASSERTM(data.dim() == 2, "data must be a 2D tensor");

    int N = data.size(0);
    int dim_origin = data.size(1);

    // Allocate output tensors.
    auto values = torch::empty({N, k}, data.options());
    auto indices = torch::empty({N, k}, torch::TensorOptions().dtype(torch::kInt32).device(data.device()));

    // Choose kernel launch parameters.
    // Each block processes WARPS_PER_BLOCK rows, with each warp (32 threads) handling one row.
    const int WARPS_PER_BLOCK = dim_origin < 1024 ? 8 : (dim_origin < 2048 ? 4 : (dim_origin < 4096 ? 2 : 1));
    const int threads = WARPS_PER_BLOCK * 32;
    const int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shared_mem_size = WARPS_PER_BLOCK * dim_origin * sizeof(float);

    // Launch the kernel based on data type.
    if (data.scalar_type() == torch::kFloat32) {
        float precision_converted = precision;
        if (WARPS_PER_BLOCK == 8) RTOPK_CALL(float, float, 8);
        else if (WARPS_PER_BLOCK == 4) RTOPK_CALL(float, float, 4);
        else if (WARPS_PER_BLOCK == 2) RTOPK_CALL(float, float, 2);
        else RTOPK_CALL(float, float, 1);
    }
#ifdef __CUDA_BF16_TYPES_EXIST__
    else if (data.scalar_type() == torch::kBFloat16) {
        __nv_bfloat16 precision_converted = __float2bfloat16(precision);
        if (WARPS_PER_BLOCK == 8) RTOPK_CALL(__nv_bfloat16, at::BFloat16, 8);
        else if (WARPS_PER_BLOCK == 4) RTOPK_CALL(__nv_bfloat16, at::BFloat16, 4);
        else if (WARPS_PER_BLOCK == 2) RTOPK_CALL(__nv_bfloat16, at::BFloat16, 2);
        else RTOPK_CALL(__nv_bfloat16, at::BFloat16, 1);
    }
#endif
    else {
        throw std::invalid_argument("Unsupported data type. Only float32 and bfloat16 are supported.");
    }

    // Check for any kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }

    return std::make_tuple(values, indices);
}

std::tuple<torch::Tensor, torch::Tensor> rtopk_forward(
    torch::Tensor data, int k, int max_iter = 10, float precision = 1e-5) 
{
    return rtopk_forward_cuda(data, k, max_iter, precision);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rtopk_forward", &rtopk_forward, "Approximate TopK forward");
}