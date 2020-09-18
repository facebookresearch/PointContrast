from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension('PG_OP', [
            'src/bfs_cluster.cpp',
            'src/bfs_cluster_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
