Parallel 1D Radix-2 FFT and Convolution
================

This repository contains an implementation of the radix-2 FFT for
one-dimensional data. It contains code for both complex-to-complex
and real-to-complex operations. The main motivation is to provide support
for the parallelisation of audio processing operations in devices such as GPUs,
CPUs and accellerators using OpenCL and Metal.

The repository includes a complete implementation of fast partitioned
convolution, based on the FFT code, as well as direct time-domain
convolution. Csound opcodes for linear time-invariant and time-varying
convolution, as well as complex-to-complex and
real-to-complex/complex-to-real FFTs are also provided.

Victor Lazzarini, 2019   
Updated for Csound 7, March 2025
Added Metal port, July 2026
******
