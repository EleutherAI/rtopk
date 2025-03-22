#include "rtopk_kernel.cuh"
#include <stdio.h>

#define EARLY_STOP

#ifdef __CUDA_BF16_TYPES_EXIST__
#define CUDART_MAX_NORMAL_BF16 __ushort_as_bfloat16((unsigned short)0x7F7FU)
#define CUDART_MIN_DENORM_BF16 __ushort_as_bfloat16((unsigned short)0x0001U)
#endif

// A helper for absolute difference (works for both float and __nv_bfloat16)
template <typename T>
__device__ inline T abs_diff(T a, T b) {
    return (a > b) ? (a - b) : (b - a);
}

// Generic max function
template <typename T>
__device__ inline T cuda_max(T a, T b) {
    return a > b ? a : b;
}

// Generic min function
template <typename T>
__device__ inline T cuda_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
__device__ inline T convert_to(float val) {
    return T(val);
}

#ifdef __CUDA_BF16_TYPES_EXIST__
// Specialization for __nv_bfloat16
template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hgt(a, b) ? a : b;
}

template <>
__device__ inline __nv_bfloat16 cuda_min(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hlt(a, b) ? a : b;
}

template <>
__device__ inline __nv_bfloat16 convert_to(float val) {
    return __float2bfloat16(val);
}
#endif

template <typename DataT, int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(DataT *data, DataT *value, int *index, int N, int dim_origin, int k, int max_iter, DataT precision)
{
    // extern __shared__ DataT cache[]; 
    // https://stackoverflow.com/a/27570775
    extern __shared__ __align__(sizeof(DataT)) unsigned char _smem[];
    DataT *cache = reinterpret_cast<DataT*>(_smem);
    
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N) {
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    // Load data into shared memory.
    #pragma unroll
    for (int ext = 0; ext < dim_len; ext++) {
        cache[wid * dim_origin + laneid + ext * 32] =
            data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    // Initialize reduction variables as DataT.
    DataT max_data, min_data;
    #ifdef __CUDA_BF16_TYPES_EXIST__
    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        max_data = CUDART_MIN_DENORM_BF16;
        min_data = CUDART_MAX_NORMAL_BF16;
    }
    else
    #endif
    {
        max_data = DataT(std::numeric_limits<DataT>::lowest());
        min_data = DataT(std::numeric_limits<DataT>::max());
    }



    #pragma unroll
    for (int j = 0; j < dim_len; j++) {
        DataT val = cache[wid * dim_origin + laneid + j * 32];
        if (val > max_data) {
            max_data = val;
        }
        if (val < min_data) {
            min_data = val;
        }
    }

    // Reduce within the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_data = cuda_max(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = cuda_min(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    DataT mid_data = max_data;
    int count;

    for (int i = 0; ; i++) {
        count = 0;
        #pragma unroll
        for (int j = 0; j < dim_len; j++) {
            DataT val = cache[wid * dim_origin + laneid + j * 32];
            count += (val >= mid_data);
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if (i >= max_iter) {
            break;
        }
#endif

        if (count < k) {
            max_data = mid_data;
        } else if (count > k) {
            min_data = mid_data;
        } else {
            break;
        }

        DataT new_mid = (min_data + max_data) / convert_to<DataT>(2.0f);
        if (new_mid <= (min_data + precision) || abs_diff(mid_data, new_mid) <= precision) {
            break;
        } else {
            mid_data = new_mid;
        }
    }

    int eq_n = k - count;
    int total_cnt = 0, total_cnt_eq = 0, total_cnt_whole = 0;

    #pragma unroll
    for (int ext = 0; ext < dim_len; ext++) {
        if (total_cnt_whole >= k) {
            break;
        }
        DataT val = cache[wid * dim_origin + laneid + ext * 32];
        bool choose = (val >= mid_data);
        bool choose_eq = (val >= min_data && val < mid_data);

        unsigned mask = __ballot_sync(0xffffffff, choose);
        unsigned mask_eq = __ballot_sync(0xffffffff, choose_eq);

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));
        int lane_cnt_eq = __popc(mask_eq & ((1 << (laneid + 1)) - 1));

        if (total_cnt + lane_cnt > k) {
            choose = false;
        }
        if (total_cnt_eq + lane_cnt_eq > eq_n) {
            choose_eq = false;
        }

        mask = __ballot_sync(0xffffffff, choose);
        mask_eq = __ballot_sync(0xffffffff, choose_eq);

        unsigned mask_whole = mask | mask_eq;

        lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));
        lane_cnt_eq = __popc(mask_eq & ((1 << (laneid + 1)) - 1));
        int lane_cnt_whole = __popc(mask_whole & ((1 << (laneid + 1)) - 1));

        if (choose || choose_eq) {
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt_whole + lane_cnt_whole - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt_whole + lane_cnt_whole - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);

        total_cnt_eq += lane_cnt_eq;
        total_cnt_eq = __shfl_sync(0xffffffff, total_cnt_eq, 31);

        total_cnt_whole += lane_cnt_whole;
        total_cnt_whole = __shfl_sync(0xffffffff, total_cnt_whole, 31);
    }
}

// Instantiate the kernel for float.
template __global__ void rtopk_kernel<float, 8>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<float, 4>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<float, 2>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<float, 1>(float*, float*, int*, int, int, int, int, float);

#ifdef __CUDA_BF16_TYPES_EXIST__
// And instantiate for bfloat16.
template __global__ void rtopk_kernel<__nv_bfloat16, 8>(__nv_bfloat16*, __nv_bfloat16*, int*, int, int, int, int, __nv_bfloat16);
template __global__ void rtopk_kernel<__nv_bfloat16, 4>(__nv_bfloat16*, __nv_bfloat16*, int*, int, int, int, int, __nv_bfloat16);
template __global__ void rtopk_kernel<__nv_bfloat16, 2>(__nv_bfloat16*, __nv_bfloat16*, int*, int, int, int, int, __nv_bfloat16);
template __global__ void rtopk_kernel<__nv_bfloat16, 1>(__nv_bfloat16*, __nv_bfloat16*, int*, int, int, int, int, __nv_bfloat16);
#endif