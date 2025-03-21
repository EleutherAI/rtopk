from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rtopk",
    ext_modules=[
        CUDAExtension(
            name="rtopk_cuda",
            sources=["rtopk_cuda.cu", "rtopk_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"]
            }
        )
    ],
    packages=["rtopk"],
    cmdclass={
        "build_ext": BuildExtension
    }
)