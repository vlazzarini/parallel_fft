/////////////////////////////////////////////////////////////////////
// Metal Partitioned Convolution class (C++ API)
// Port of cl_conv.h to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __MT_CONV_H__
#define __MT_CONV_H__
#include <complex>
#include <iostream>
#include <string>
#include <Metal/Metal.hpp>

namespace mt_conv {

class Mtpconv {
  int N, bins;
  int bsize, nparts, wp, wp2;
  MTL::Buffer *w[2], *w2[2], *b;
  MTL::Buffer *in1, *in2, *out;
  MTL::Buffer *spec1, *spec2, *olap;
  MTL::Device *device;
  MTL::CommandQueue *commands1, *commands2;
  MTL::ComputePipelineState *reorder_pso, *fft_pso;
  MTL::ComputePipelineState *r2c_pso, *c2r_pso;
  MTL::ComputePipelineState *convol_pso, *olap_pso;
  void (*err)(std::string s, void *uData);
  void *userData;
  int mt_err;
  int host_mem;

  static void msg(std::string str, void *userData) {
    if (userData == NULL)
      std::cout << str << std::endl;
  }

public:
  /** Constructor \n
      device - Metal device \n
      cvs - impulse response size \n
      pts - partition size \n
      errs - error message callback \n
      uData - callback user data \n
      in1, in2, out - host-allocated arrays (optional)
  */
  Mtpconv(MTL::Device *device, int cvs, int pts,
          void (*errs)(std::string s, void *d) = NULL, void *uData = NULL,
          void *in1 = NULL, void *in2 = NULL, void *out = 0);
  ~Mtpconv();

  /** set the convolution impulse response
      ir - impulse response of size cvs;
  */
  int push_ir(float *ir);

  /** Convolution computation
      output - output array (partition-size samples) \n
      input - input array   \n
  */
  int convolution(float *output, float *input);

  /** Time-varying convolution computation
      output - output array (partition size samples) \n
      input1, input2 - input arrays \n
      NB: if input arrays are held by the host, they
      need to have 2*partition size floats, but
      only filled with partition-size samples   \n
  */
  int convolution(float *output, float *input1, float *input2);

  /** get a recorded error code, 0 if no error was recorded
   */
  int get_mt_err() { return mt_err; }

  /** this needs to be called after convolution to
      synchronise threads */
  void synchronise() {
    MTL::CommandBuffer *cb = commands1->commandBuffer();
    cb->commit();
    cb->waitUntilCompleted();
  }
};
}
#endif
