from setuptools import *

libTM = Extension('libTM',
                  sources = ['pyTsetlinMachineParallel/ConvolutionalTsetlinMachine.c', 'pyTsetlinMachineParallel/MultiClassConvolutionalTsetlinMachine.c', 'pyTsetlinMachineParallel/Tools.c'],
                  include_dirs=['pyTsetlinMachineParallel'],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-lgomp'])

setup(
   name='pyTsetlinMachineParallel',
   version='1.1.0',
   authors=['Ole-Christoffer Granmo', 'Calvin Kinateder'],
   author_emails=['ole.granmo@uia.no', 'calvin.kinateder@gmail.com'],
   url='https://github.com/ckinateder/pyTsetlinMachineParallel/',
   license='MIT',
   description='Multi-threaded implementation of the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and multigranularity.',
   long_description='Multi-threaded implementation of the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and multigranularity.',
   ext_modules = [libTM],
   keywords ='pattern-recognition machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic tsetlin-machine regression convolution',
   packages=['pyTsetlinMachineParallel']
)
