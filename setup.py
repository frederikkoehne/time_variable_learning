from setuptools import setup, find_packages

print(find_packages())

setup(
    name='tau_modules',
    version='1.0.0',
    author='Frederik Köhne',
    author_email='frederik.koehne@uni-bayreuth.de',
    description='Implementation of time variable neural network architectures',
    packages=find_packages(),
    install_requires=[
        'torch>=1.13.0',
    ],
)
