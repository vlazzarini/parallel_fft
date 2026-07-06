/////////////////////////////////////////////////////////////////////
// OpenCL 1-D Radix-2 FFT classes
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __CL_FFT_H__
#define __CL_FFT_H__

#include <complex>
#ifdef __MACH__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <iostream>

namespace cl_fft {

const double PI = 3.141592653589793;
const char *cl_error_string(int err);

/** Complex to Complex FFT class
 **/
class Clcfft {

protected:
  int N;
  bool forward;
  cl_mem w, b, data1, data2;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel fft_kernel, reorder_kernel;
  size_t wgs, rwgs;
  char log[2048];
  size_t llen;
  int cl_err;

  int fft();

public:
  /** Constructor \n
      device_id - OpenCL device ID \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
  */
  Clcfft(cl_device_id device_id, int size, bool fwd = true);

  /** Destructor
   */
  virtual ~Clcfft();

  /** DFT operation (in-place) \n
      c - data array with N complex numbers \n
  */
  virtual int transform(std::complex<float> *c);

  /** Get setup error code
   */
  int get_error() { return cl_err; }

  /** Get compilation log
   */
  const char *get_log() { return (const char *)log; }
};

/** Real to Complex FFT class
 **/
class Clrfft : public Clcfft {

  cl_mem w2;
  cl_kernel conv_kernel, iconv_kernel;
  size_t cwgs, iwgs;

public:
  /** Constructor \n
      device_id - OpenCL device ID \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
  */
  Clrfft(cl_device_id device_id, int size, bool fwd);

  /** Destructor
   */
  virtual ~Clrfft();

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
    int err;
    float *r = reinterpret_cast<float *>(c);
    err = transform(c, r);
    return err;
  }
};
}

#endif
