/////////////////////////////////////////////////////////////////////
// Metal 1-D Radix-2 FFT test program - complex to complex (C++ API)
// Port of test_cfft.cpp to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "mt_fft.h"
#include <iomanip>
#include <iostream>
#include <vector>

#define N 16

using namespace mt_fft;

int main() {
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    std::cout << "Metal is not supported on this device" << std::endl;
    return -1;
  }
  std::cout << "using device: " << device->name()->utf8String() << std::endl;

  Mtcfft dft(device, N, true);
  if (dft.get_error()) {
    std::cout << "error setting up forward FFT" << std::endl;
    return 1;
  }

  Mtcfft idft(device, N, false);
  if (idft.get_error()) {
    std::cout << "error setting up inverse FFT" << std::endl;
    return 1;
  }

  std::vector<std::complex<float>> sig(N);
  for (int i = 0; i < N; i++)
    sig[i].real(sin(i * 2 * PI / N));

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "in =[";
  for (int i = 0; i < N - 1; i++)
    std::cout << sig[i].real() << ", ";
  std::cout << sig[N - 1].real() << "]" << std::endl;

  dft.transform(sig.data());

  std::cout << "spec =[";
  for (int i = 0; i < N - 1; i++)
    std::cout << sig[i] << ",";
  std::cout << sig[N - 1] << "]" << std::endl;

  idft.transform(sig.data());

  std::cout << "out =[";
  for (int i = 0; i < N - 1; i++)
    std::cout << sig[i].real() << ",";
  std::cout << sig[N - 1].real() << "]" << std::endl;

  device->release();
  return 0;
}
