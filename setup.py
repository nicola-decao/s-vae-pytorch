
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='hyperspherical_vae',
    version='0.1.1',
    author='Nicola De Cao, Tim R. Davidson, Luca Falorsi',
    author_email='nicola.decao@gmail.com',
    description='Pytorch implementation of Hyperspherical Variational Auto-Encoders',
    license='MIT',
    keywords='pytorch vae variational-auto-encoder von-mises-fisher  machine-learning deep-learning manifold-learning',
    url='https://nicola-decao.github.io/s-vae-pytorch/',
    download_url='https://github.com/nicola-decao/SVAE',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    install_requires=['numpy', 'torch>=0.4.1', 'scipy>=1.0.0', 'numpy'],
    packages=find_packages()
)
