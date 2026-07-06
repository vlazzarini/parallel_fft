/////////////////////////////////////////////////////////////////////
// Metal Direct Convolution class (C++ API)
// Port of cl_dconv.h to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __MT_DCONV_H__
#define __MT_DCONV_H__
#include <iostream>
#include <string>
#include <Metal/Metal.hpp>

namespace mt_conv {

class Mtdconv {
  int irsize, vsize, wp;
  MTL::Buffer *buff, *coefs, *del;
  MTL::Device *device;
  MTL::CommandQueue *commands;
  MTL::ComputePipelineState *convol_pso;
  void (*err)(std::string s, void *uData);
  void *userData;
  int mt_err;

  static void msg(std::string str, void *userData) {
    if (userData == NULL)
      std::cout << str << std::endl;
  }

public:
  /** Constructor \n
      device - Metal device \n
      cvs - impulse response size \n
      vsize - processing vector size \n
      errs - error message callback \n
      uData - callback user data \n
  */
  Mtdconv(MTL::Device *device, int cvs, int vsize,
          void (*errs)(std::string s, void *d) = NULL, void *uData = NULL);

  ~Mtdconv();

  /** set the convolution impulse response
      ir - impulse response of size cvs;
  */
  int push_ir(float *ir);

  /** Convolution computation
      output - output array (vsize samples) \n
      input - input array (vsize samples) \n
  */
  int convolution(float *output, float *input);

  int convolution(float *out, float *in1, float *in2);

  /** get a recorded error code, 0 if no error was recorded
   */
  int get_mt_err() { return mt_err; }
};
}
#endif
