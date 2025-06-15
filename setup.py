from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='backbone_3d',
    version='1.0.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src', include=['backbone_3d*']),
    ext_modules=[
        CUDAExtension(
            name='backbone_3d.ext',
            sources=[
                'src/backbone_3d/extensions/extra/cloud/cloud.cpp',
                'src/backbone_3d/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'src/backbone_3d/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'src/backbone_3d/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'src/backbone_3d/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'src/backbone_3d/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)