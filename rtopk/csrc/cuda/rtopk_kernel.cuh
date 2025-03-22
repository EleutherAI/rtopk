#pragma once
#include <cuda_bf16.h>

template <typename DataT, int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(DataT *data, DataT *value, int *index, int N, int dim_origin, int k, int max_iter, DataT precision);
