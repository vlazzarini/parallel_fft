#!/bin/sh
set -x
c++ -O3 -std=c++14 -o clrfft cl_fft.cpp test_rfft.cpp -I. -framework OpenCL -Wno-deprecated-declarations
c++ -O3 -std=c++14 -o clcfft cl_fft.cpp test_cfft.cpp -I. -framework OpenCL -Wno-deprecated-declarations
c++ -O3 -std=c++14 -dynamiclib -o libcl_conv.dylib -DUSE_DOUBLE -D_FORTIFY_SOURCE=0 cl_conv.cpp cl_dconv.cpp -framework OpenCL -Wno-deprecated-declarations

