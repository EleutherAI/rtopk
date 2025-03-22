cmake_minimum_required(VERSION 3.5)
project (rtopk_test C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH ".venv/lib/python3.10/site-packages/torch")

#This is needed to be able to find the mkl and other dependent libraries
link_directories(".venv/lib")
set(ENV{MKLROOT} ".venv/lib")

find_package(Torch REQUIRED CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(${PROJECT_NAME} rtopk/csrc/cuda/rtopk_kernel.cu test.cu)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

target_link_libraries(${PROJECT_NAME}  curand stdc++fs)