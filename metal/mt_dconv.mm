/////////////////////////////////////////////////////////////////////
// Metal Direct Convolution implementation (C++ API)
// Port of cl_dconv.cpp to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "mt_dconv.h"

namespace mt_conv {

const char *dconv_msl = R"(
inline void atomic_add_f(device float *source, const float operand) {
  union { uint intVal; float floatVal; } newVal, prevVal;
  prevVal.floatVal = *source;
  do {
    newVal.floatVal = prevVal.floatVal + operand;
  } while (!atomic_compare_exchange_weak(
    (device atomic_uint *)source, &prevVal.intVal, newVal.intVal));
}

kernel void convol(device float *out [[buffer(0)]],
                   device const float *del [[buffer(1)]],
                   device const float *coefs [[buffer(2)]],
                   constant int &irsize [[buffer(3)]],
                   constant int &rp [[buffer(4)]],
                   constant int &vsize [[buffer(5)]],
                   uint id [[thread_position_in_grid]]) {
  int t = id;
  if (t >= irsize * vsize) return;
  int n = t % vsize;
  int h = t / vsize;
  int end = irsize + vsize;
  int rp_adj = rp + n + h;
  float tap = del[rp_adj < end ? rp_adj : rp_adj % end] * coefs[irsize - 1 - h];
  atomic_add_f(&out[n], tap);
}
)";

Mtdconv::Mtdconv(MTL::Device *dev, int cvs, int vsiz,
                 void (*errs)(std::string s, void *d), void *uData)
    : irsize(cvs), vsize(vsiz), wp(0), buff(nullptr), coefs(nullptr),
      del(nullptr), device(dev), commands(nullptr), convol_pso(nullptr),
      err(errs == nullptr ? this->msg : errs), userData(uData),
      mt_err(0) {

  device->retain();

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  commands = device->newCommandQueue();
  if (!commands) {
    err("error creating Metal command queue\n", userData);
    mt_err = -1;
    pool->release();
    return;
  }

  NS::Error *error = nullptr;
  MTL::Library *lib = device->newLibrary(
      NS::String::string(dconv_msl, NS::UTF8StringEncoding), nullptr, &error);
  if (!lib) {
    err("error creating Metal library\n", userData);
    mt_err = -2;
    pool->release();
    return;
  }

  MTL::Function *fn = lib->newFunction(
      NS::String::string("convol", NS::UTF8StringEncoding));
  convol_pso = device->newComputePipelineState(fn, &error);
  fn->release();
  lib->release();

  if (!convol_pso) {
    err("error creating Metal pipeline state\n", userData);
    mt_err = -3;
    pool->release();
    return;
  }

  MTL::ResourceOptions opts = MTL::ResourceStorageModeShared;
  buff = device->newBuffer(vsize * sizeof(float), opts);
  del = device->newBuffer((irsize + vsize) * sizeof(float), opts);
  coefs = device->newBuffer((irsize + vsize) * sizeof(float), opts);

  pool->release();
}

Mtdconv::~Mtdconv() {
  if (del) del->release();
  if (buff) buff->release();
  if (coefs) coefs->release();
  if (convol_pso) convol_pso->release();
  if (commands) commands->release();
  if (device) device->release();
}

int Mtdconv::convolution(float *out, float *in) {
  size_t bytes = vsize * sizeof(float);
  size_t threads = irsize * vsize;

  if (wp > irsize) {
    int front = wp - irsize;
    bytes = (vsize - front) * sizeof(float);
    memcpy((uint8_t *)del->contents() + wp * sizeof(float), in, bytes);
    bytes = front * sizeof(float);
    memcpy(del->contents(), &in[vsize - front], bytes);
    bytes = vsize * sizeof(float);
  } else {
    memcpy((uint8_t *)del->contents() + wp * sizeof(float), in, bytes);
  }

  memset(buff->contents(), 0, bytes);

  wp = (wp + vsize) % (irsize + vsize);

  MTL::CommandBuffer *cmdBuf = commands->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();

  enc->setComputePipelineState(convol_pso);
  enc->setBuffer(buff, 0, 0);
  enc->setBuffer(del, 0, 1);
  enc->setBuffer(coefs, 0, 2);
  enc->setBytes(&irsize, sizeof(int), 3);
  enc->setBytes(&wp, sizeof(int), 4);
  enc->setBytes(&vsize, sizeof(int), 5);

  NS::UInteger tgSize = convol_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > threads) tgSize = threads;
  enc->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  memcpy(out, buff->contents(), vsize * sizeof(float));
  return 0;
}

int Mtdconv::convolution(float *out, float *in1, float *in2) {
  size_t bytes = vsize * sizeof(float);
  if (wp > irsize) {
    int front = wp - irsize;
    bytes = (vsize - front) * sizeof(float);
    memcpy((uint8_t *)coefs->contents() + wp * sizeof(float), in2, bytes);
    bytes = front * sizeof(float);
    memcpy(coefs->contents(), &in2[vsize - front], bytes);
    bytes = vsize * sizeof(float);
  } else {
    memcpy((uint8_t *)coefs->contents() + wp * sizeof(float), in2, bytes);
  }
  return convolution(out, in1);
}

int Mtdconv::push_ir(float *ir) {
  memcpy(coefs->contents(), ir, irsize * sizeof(float));
  return 0;
}
}
