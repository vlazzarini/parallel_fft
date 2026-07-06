/////////////////////////////////////////////////////////////////////
// Metal 1-D Radix-2 FFT classes (C++ API)
// Port of cl_fft.h to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __MT_FFT_H__
#define __MT_FFT_H__

#include <complex>
#include <Metal/Metal.hpp>

namespace mt_fft {

const double PI = 3.141592653589793;

/** Complex to Complex FFT class
 **/
class Mtcfft {

protected:
  int N;
  bool forward;
  MTL::Buffer *w, *b, *data1, *data2;
  MTL::Device *device;
  MTL::CommandQueue *commands;
  MTL::ComputePipelineState *fft_pso, *reorder_pso;
  int mt_err;

  int fft(MTL::CommandBuffer *cmdBuf);

public:
  /** Constructor \n
      device - Metal device \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
  */
  Mtcfft(MTL::Device *device, int size, bool fwd = true);

  /** Destructor
   */
  virtual ~Mtcfft();

  /** DFT operation (in-place) \n
      c - data array with N complex numbers \n
  */
  virtual int transform(std::complex<float> *c);

  /** Get setup error code
   */
  int get_error() { return mt_err; }
};

/** Real to Complex FFT class
 **/
class Mtrfft : public Mtcfft {

  MTL::Buffer *w2;
  MTL::ComputePipelineState *conv_pso, *iconv_pso;

public:
  /** Constructor \n
      device - Metal device \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
  */
  Mtrfft(MTL::Device *device, int size, bool fwd);

  /** Destructor
   */
  virtual ~Mtrfft();

  /** DFT operation (out-of-place or in-place) \n
      c - data array with N/2 complex numbers \n
      r - data array with N real numbers \n
      Transform is in place if both c and r point to the same memory.\n
      If separate locations are used, r holds input data in forward transform \n
      and c will contain the output. For inverse, c is input, r is output. \n
  */
  int transform(std::complex<float> *c, float *r);

  /** DFT operation (in-place) \n
      c - data array (N real points or N/2 complex points, encoded as a \n
      complex array) \n
  */
  virtual int transform(std::complex<float> *c) {
    float *r = reinterpret_cast<float *>(c);
    return transform(c, r);
  }
};
}

#endif
