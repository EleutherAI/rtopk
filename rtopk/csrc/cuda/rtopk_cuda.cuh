#pragma once
#include <Python.h>
#include <ATen/Operators.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>
#include "rtopk_kernel.cuh"


std::tuple<at::Tensor, at::Tensor> rtopk_forward_cuda(
    at::Tensor data,
    int64_t k,
    int64_t max_iter = 10,
    double precision = 1e-5);