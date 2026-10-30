from setuptools import setup,Extension
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'build_deps'))
import pybind11
setup(name='hnsw_observer',version='0.1',ext_modules=[Extension('hnsw_observer',['observer.cpp'],include_dirs=[pybind11.get_include(),'.'],language='c++',extra_compile_args=['/O2','/EHsc','/std:c++17'] if sys.platform=='win32' else ['-O3','-std=c++17','-march=native'])])
