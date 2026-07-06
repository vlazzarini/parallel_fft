////////////////////////////////////////////////////////////////////////////////
// Metal Partitioned Convolution implementation (C++ API)
// Port of cl_conv.cpp to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
////////////////////////////////////////////////////////////////////////////////
#include "mt_conv.h"
#include <vector>
#include <cmath>

namespace mt_conv {

const double PI = M_PI;

const char *pconv_msl = R"(
typedef float2 cmplx;

inline void atomic_add_f(device float *source, const float operand) {
  union { uint intVal; float floatVal; } newVal, prevVal;
  prevVal.floatVal = *source;
  do {
    newVal.floatVal = prevVal.floatVal + operand;
  } while (!atomic_compare_exchange_weak(
    (device atomic_uint *)source, &prevVal.intVal, newVal.intVal));
}

kernel void reorder(device cmplx *out [[buffer(0)]],
                    device cmplx *in [[buffer(1)]],
                    constant int *b [[buffer(2)]],
                    constant int &offs [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  out += offs;
  out[id] = in[b[id]];
  in[b[id]] = 0.f;
}

kernel void fft(device cmplx *s [[buffer(0)]],
                device cmplx *w [[buffer(1)]],
                constant int &N [[buffer(2)]],
                constant int &n2 [[buffer(3)]],
                constant int &offs [[buffer(4)]],
                uint id [[thread_position_in_grid]]) {
  s += offs;
  int k = id * n2;
  int m = k / N;
  int n = n2 >> 1;
  k = k % N + m;
  int i = k + n;
  float er = (s + k)->x, ei = (s + k)->y;
  float sr = (s + i)->x, si = (s + i)->y;
  float wr = (w + m * N / n2)->x, wi = (w + m * N / n2)->y;
  float pr = sr * wr - si * wi;
  float pi = sr * wi + si * wr;
  (s + k)->x = er + pr; (s + k)->y = ei + pi;
  (s + i)->x = er - pr; (s + i)->y = ei - pi;
}

kernel void r2c(device cmplx *c [[buffer(0)]],
                device cmplx *w [[buffer(1)]],
                constant int &N [[buffer(2)]],
                constant int &offs [[buffer(3)]],
                uint id [[thread_position_in_grid]]) {
  int i = id;
  c += offs;
  if (!i) {
    float c0x = (c + 0)->x, c0y = (c + 0)->y;
    (c + 0)->x = (c0x + c0y) * 0.5f; (c + 0)->y = (c0x - c0y) * 0.5f;
    return;
  }
  int j = N - i;
  float cjx = (c + j)->x, cjy = (c + j)->y;
  float cix = (c + i)->x, ciy = (c + i)->y;
  float ex = (cix + cjx) * 0.5f, ey = (ciy - cjy) * 0.5f;
  float ox = (cjy + ciy) * 0.5f, oy = (cjx - cix) * 0.5f;
  float wr = (w + i)->x, wi = (w + i)->y;
  float px = ox * wr - oy * wi;
  float py = ox * wi + oy * wr;
  (c + i)->x = ex + px; (c + i)->y = ey + py;
  (c + j)->x = ex - px; (c + j)->y = -(ey - py);
}

kernel void c2r(device cmplx *c [[buffer(0)]],
                device cmplx *w [[buffer(1)]],
                constant int &N [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
  if (!id) {
    float c0x = (c + 0)->x, c0y = (c + 0)->y;
    (c + 0)->x = (c0x + c0y); (c + 0)->y = (c0x - c0y);
    return;
  }
  int j = N - id;
  float cjx = (c + j)->x, cjy = (c + j)->y;
  float cidx = (c + id)->x, cidy = (c + id)->y;
  float ex = (cidx + cjx) * 0.5f, ey = (cidy - cjy) * 0.5f;
  float ox = (-cidy - cjy) * 0.5f, oy = (cidx - cjx) * 0.5f;
  float wr = (w + id)->x, wi = (w + id)->y;
  float px = ox * wr - oy * wi;
  float py = ox * wi + oy * wr;
  (c + id)->x = ex + px; (c + id)->y = ey + py;
  (c + j)->x = ex - px; (c + j)->y = -(ey - py);
}

kernel void convol(device float *out [[buffer(0)]],
                   device const cmplx *in [[buffer(1)]],
                   device const cmplx *coef [[buffer(2)]],
                   constant int &rp [[buffer(3)]],
                   constant int &b [[buffer(4)]],
                   constant int &nparts [[buffer(5)]],
                   uint id [[thread_position_in_grid]]) {
  int k = id;
  int n = k % b;
  int n2 = n << 1;
  cmplx s;
  int rp_adj = rp + k / b;
  in += (rp_adj < nparts ? rp_adj : rp_adj - nparts) * b;
  if (n) {
    float inr = (in + n)->x, ini = (in + n)->y;
    float ckr = (coef + k)->x, cki = (coef + k)->y;
    s = (cmplx)(inr * ckr - ini * cki, inr * cki + ini * ckr);
  } else {
    s = (cmplx)((in + 0)->x * (coef + k)->x, (in + 0)->y * (coef + k)->y);
  }
  atomic_add_f(&out[n2], s.x);
  atomic_add_f(&out[n2 + 1], s.y);
}

kernel void olap(device float *buf [[buffer(0)]],
                 device const float *in [[buffer(1)]],
                 constant int &parts [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
  int n = id;
  buf[n] = (in[n] + buf[parts + n]) / parts;
  buf[parts + n] = in[parts + n];
}
)";

Mtpconv::Mtpconv(MTL::Device *dev, int cvs, int pts,
                 void (*errs)(std::string s, void *d), void *uData,
                 void *inp1, void *inp2, void *outp)
    : N(pts << 1), bins(pts), bsize((cvs / pts) * bins), nparts(cvs / pts),
      wp(0), wp2(nparts - 1), w{nullptr, nullptr}, w2{nullptr, nullptr},
      b(nullptr), in1(nullptr), in2(nullptr), out(nullptr), spec1(nullptr),
      spec2(nullptr), olap(nullptr), device(dev), commands1(nullptr),
      commands2(nullptr), reorder_pso(nullptr), fft_pso(nullptr),
      r2c_pso(nullptr), c2r_pso(nullptr), convol_pso(nullptr),
      olap_pso(nullptr), err(errs == nullptr ? this->msg : errs),
      userData(uData), mt_err(0),
      host_mem(((uintptr_t)inp1 & (uintptr_t)inp2 & (uintptr_t)outp) ? 1 : 0) {

  device->retain();

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  commands1 = device->newCommandQueue();
  commands2 = device->newCommandQueue();
  if (!commands1 || !commands2) {
    err("error creating Metal command queues\n", userData);
    mt_err = -1;
    pool->release();
    return;
  }

  NS::Error *error = nullptr;
  MTL::Library *lib = device->newLibrary(
      NS::String::string(pconv_msl, NS::UTF8StringEncoding), nullptr, &error);
  if (!lib) {
    err("error creating Metal library\n", userData);
    mt_err = -2;
    pool->release();
    return;
  }

  auto nsstr = [](const char *s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
  };

  MTL::Function *reorder_fn = lib->newFunction(nsstr("reorder"));
  MTL::Function *fft_fn = lib->newFunction(nsstr("fft"));
  MTL::Function *r2c_fn = lib->newFunction(nsstr("r2c"));
  MTL::Function *c2r_fn = lib->newFunction(nsstr("c2r"));
  MTL::Function *convol_fn = lib->newFunction(nsstr("convol"));
  MTL::Function *olap_fn = lib->newFunction(nsstr("olap"));

  reorder_pso = device->newComputePipelineState(reorder_fn, &error);
  fft_pso = device->newComputePipelineState(fft_fn, &error);
  r2c_pso = device->newComputePipelineState(r2c_fn, &error);
  c2r_pso = device->newComputePipelineState(c2r_fn, &error);
  convol_pso = device->newComputePipelineState(convol_fn, &error);
  olap_pso = device->newComputePipelineState(olap_fn, &error);

  reorder_fn->release();
  fft_fn->release();
  r2c_fn->release();
  c2r_fn->release();
  convol_fn->release();
  olap_fn->release();
  lib->release();

  if (!reorder_pso || !fft_pso || !r2c_pso || !c2r_pso || !convol_pso || !olap_pso) {
    err("error creating Metal pipeline states\n", userData);
    mt_err = -3;
    pool->release();
    return;
  }

  MTL::ResourceOptions opts = MTL::ResourceStorageModeShared;
  size_t bufSize = bins * sizeof(float) * 2;

  in1 = inp1 ? device->newBuffer(inp1, bufSize, opts) :
               device->newBuffer(bufSize, opts);
  in2 = inp2 ? device->newBuffer(inp2, bufSize, opts) :
               device->newBuffer(bufSize, opts);
  olap = outp ? device->newBuffer(outp, bufSize, opts) :
               device->newBuffer(bufSize, opts);

  out = device->newBuffer(bufSize, opts);
  spec1 = device->newBuffer(bsize * sizeof(float) * 2, opts);
  spec2 = device->newBuffer(bsize * sizeof(float) * 2, opts);

  w[0] = device->newBuffer(bufSize, opts);
  w[1] = device->newBuffer(bufSize, opts);
  w2[0] = device->newBuffer(bufSize, opts);
  w2[1] = device->newBuffer(bufSize, opts);
  b = device->newBuffer(bins * sizeof(int), opts);

  if (!in1 || !in2 || !olap || !out || !spec1 || !spec2 ||
      !w[0] || !w[1] || !w2[0] || !w2[1] || !b) {
    err("error creating Metal buffers\n", userData);
    mt_err = -4;
    pool->release();
    return;
  }

  float *wd = (float *)w[0]->contents();
  for (int i = 0; i < bins; i++) {
    wd[i * 2] = cos(i * 2 * PI / bins);
    wd[i * 2 + 1] = -sin(i * 2 * PI / bins);
  }
  wd = (float *)w[1]->contents();
  for (int i = 0; i < bins; i++) {
    wd[i * 2] = cos(i * 2 * PI / bins);
    wd[i * 2 + 1] = sin(i * 2 * PI / bins);
  }
  wd = (float *)w2[0]->contents();
  for (int i = 0; i < bins; i++) {
    wd[i * 2] = cos(i * PI / bins);
    wd[i * 2 + 1] = -sin(i * PI / bins);
  }
  wd = (float *)w2[1]->contents();
  for (int i = 0; i < bins; i++) {
    wd[i * 2] = cos(i * PI / bins);
    wd[i * 2 + 1] = sin(i * PI / bins);
  }

  int *bp = (int *)b->contents();
  for (int i = 0; i < bins; i++)
    bp[i] = i;
  for (int i = 1, n = bins / 2; i < bins; i = i << 1, n = n >> 1)
    for (int j = 0; j < i; j++)
      bp[i + j] = bp[j] + n;

  memset(olap->contents(), 0, bins * sizeof(float) * 2);
  memset(spec1->contents(), 0, bsize * sizeof(float) * 2);
  memset(spec2->contents(), 0, bsize * sizeof(float) * 2);
  memset(in1->contents(), 0, bins * sizeof(float) * 2);
  memset(in2->contents(), 0, bins * sizeof(float) * 2);

  pool->release();
}

Mtpconv::~Mtpconv() {
  if (w[0]) w[0]->release();
  if (w[1]) w[1]->release();
  if (w2[0]) w2[0]->release();
  if (w2[1]) w2[1]->release();
  if (b) b->release();
  if (out) out->release();
  if (in2) in2->release();
  if (in1) in1->release();
  if (olap) olap->release();
  if (spec1) spec1->release();
  if (spec2) spec2->release();
  if (fft_pso) fft_pso->release();
  if (reorder_pso) reorder_pso->release();
  if (r2c_pso) r2c_pso->release();
  if (c2r_pso) c2r_pso->release();
  if (convol_pso) convol_pso->release();
  if (olap_pso) olap_pso->release();
  if (commands1) commands1->release();
  if (commands2) commands2->release();
  if (device) device->release();
}

int Mtpconv::push_ir(float *ir) {
  size_t bytes = bins * sizeof(float) * 2;
  NS::UInteger tgSize = reorder_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > bins) tgSize = bins;

  for (int i = 0; i < nparts; i++) {
    if (!host_mem)
      memcpy(in2->contents(), &ir[i * bins], bytes >> 1);

    MTL::CommandBuffer *cmdBuf = commands1->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();

    int offs = wp2 * bins;
    enc->setComputePipelineState(reorder_pso);
    enc->setBuffer(spec2, 0, 0);
    enc->setBuffer(in2, 0, 1);
    enc->setBuffer(b, 0, 2);
    enc->setBytes(&offs, sizeof(int), 3);
    enc->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->setComputePipelineState(fft_pso);
    enc->setBuffer(spec2, 0, 0);
    enc->setBuffer(w[0], 0, 1);
    enc->setBytes(&bins, sizeof(int), 2);
    for (int n = 1; n < bins; n *= 2) {
      int n2 = n << 1;
      enc->setBytes(&n2, sizeof(int), 3);
      enc->setBytes(&offs, sizeof(int), 4);
      enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
    }

    enc->setComputePipelineState(r2c_pso);
    enc->setBuffer(spec2, 0, 0);
    enc->setBuffer(w2[0], 0, 1);
    enc->setBytes(&bins, sizeof(int), 2);
    enc->setBytes(&offs, sizeof(int), 3);
    enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;
  }
  return 0;
}

int Mtpconv::convolution(float *output, float *input) {
  size_t bytes = bins * sizeof(float) * 2;
  NS::UInteger tgSize = reorder_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > bins) tgSize = bins;

  if (!host_mem)
    memcpy(in1->contents(), input, bytes >> 1);

  MTL::CommandBuffer *cmdBuf = commands1->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();

  int offs = wp * bins;
  enc->setComputePipelineState(reorder_pso);
  enc->setBuffer(spec1, 0, 0);
  enc->setBuffer(in1, 0, 1);
  enc->setBuffer(b, 0, 2);
  enc->setBytes(&offs, sizeof(int), 3);
  enc->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->setComputePipelineState(fft_pso);
  enc->setBuffer(spec1, 0, 0);
  enc->setBuffer(w[0], 0, 1);
  enc->setBytes(&bins, sizeof(int), 2);
  for (int n = 1; n < bins; n *= 2) {
    int n2 = n << 1;
    enc->setBytes(&n2, sizeof(int), 3);
    enc->setBytes(&offs, sizeof(int), 4);
    enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  }

  enc->setComputePipelineState(r2c_pso);
  enc->setBuffer(spec1, 0, 0);
  enc->setBuffer(w2[0], 0, 1);
  enc->setBytes(&bins, sizeof(int), 2);
  enc->setBytes(&offs, sizeof(int), 3);
  enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  wp = wp != nparts - 1 ? wp + 1 : 0;

  cmdBuf = commands1->commandBuffer();
  enc = cmdBuf->computeCommandEncoder();

  enc->setComputePipelineState(convol_pso);
  enc->setBuffer(in1, 0, 0);
  enc->setBuffer(spec1, 0, 1);
  enc->setBuffer(spec2, 0, 2);
  enc->setBytes(&wp, sizeof(int), 3);
  enc->setBytes(&bins, sizeof(int), 4);
  enc->setBytes(&nparts, sizeof(int), 5);
  enc->dispatchThreads(MTL::Size(bsize, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->setComputePipelineState(c2r_pso);
  enc->setBuffer(in1, 0, 0);
  enc->setBuffer(w2[1], 0, 1);
  enc->setBytes(&bins, sizeof(int), 2);
  enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

  int zero_offs = 0;
  enc->setComputePipelineState(reorder_pso);
  enc->setBuffer(out, 0, 0);
  enc->setBuffer(in1, 0, 1);
  enc->setBuffer(b, 0, 2);
  enc->setBytes(&zero_offs, sizeof(int), 3);
  enc->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->setComputePipelineState(fft_pso);
  enc->setBuffer(out, 0, 0);
  enc->setBuffer(w[1], 0, 1);
  enc->setBytes(&bins, sizeof(int), 2);
  for (int n = 1; n < bins; n *= 2) {
    int n2 = n << 1;
    enc->setBytes(&n2, sizeof(int), 3);
    enc->setBytes(&zero_offs, sizeof(int), 4);
    enc->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  }

  enc->setComputePipelineState(olap_pso);
  enc->setBuffer(olap, 0, 0);
  enc->setBuffer(out, 0, 1);
  enc->setBytes(&bins, sizeof(int), 2);
  enc->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  if (!host_mem)
    memcpy(output, olap->contents(), bytes >> 1);

  return 0;
}

int Mtpconv::convolution(float *output, float *input1, float *input2) {
  size_t bytes = bins * sizeof(float) * 2;
  NS::UInteger tgSize = reorder_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > bins) tgSize = bins;

  if (!host_mem) {
    memcpy(in1->contents(), input1, bytes >> 1);
    memcpy(in2->contents(), input2, bytes >> 1);
  }

  MTL::CommandBuffer *cmdBuf1 = commands1->commandBuffer();
  MTL::ComputeCommandEncoder *enc1 = cmdBuf1->computeCommandEncoder();
  MTL::CommandBuffer *cmdBuf2 = commands2->commandBuffer();
  MTL::ComputeCommandEncoder *enc2 = cmdBuf2->computeCommandEncoder();

  int offs1 = wp * bins;
  int offs2 = wp2 * bins;

  enc1->setComputePipelineState(reorder_pso);
  enc1->setBuffer(spec1, 0, 0);
  enc1->setBuffer(in1, 0, 1);
  enc1->setBuffer(b, 0, 2);
  enc1->setBytes(&offs1, sizeof(int), 3);
  enc1->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc1->setComputePipelineState(fft_pso);
  enc1->setBuffer(spec1, 0, 0);
  enc1->setBuffer(w[0], 0, 1);
  enc1->setBytes(&bins, sizeof(int), 2);
  for (int n = 1; n < bins; n *= 2) {
    int n2 = n << 1;
    enc1->setBytes(&n2, sizeof(int), 3);
    enc1->setBytes(&offs1, sizeof(int), 4);
    enc1->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  }

  enc1->setComputePipelineState(r2c_pso);
  enc1->setBuffer(spec1, 0, 0);
  enc1->setBuffer(w2[0], 0, 1);
  enc1->setBytes(&bins, sizeof(int), 2);
  enc1->setBytes(&offs1, sizeof(int), 3);
  enc1->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  enc1->endEncoding();
  cmdBuf1->commit();

  enc2->setComputePipelineState(reorder_pso);
  enc2->setBuffer(spec2, 0, 0);
  enc2->setBuffer(in2, 0, 1);
  enc2->setBuffer(b, 0, 2);
  enc2->setBytes(&offs2, sizeof(int), 3);
  enc2->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc2->setComputePipelineState(fft_pso);
  enc2->setBuffer(spec2, 0, 0);
  enc2->setBuffer(w[0], 0, 1);
  enc2->setBytes(&bins, sizeof(int), 2);
  for (int n = 1; n < bins; n *= 2) {
    int n2 = n << 1;
    enc2->setBytes(&n2, sizeof(int), 3);
    enc2->setBytes(&offs2, sizeof(int), 4);
    enc2->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  }

  enc2->setComputePipelineState(r2c_pso);
  enc2->setBuffer(spec2, 0, 0);
  enc2->setBuffer(w2[0], 0, 1);
  enc2->setBytes(&bins, sizeof(int), 2);
  enc2->setBytes(&offs2, sizeof(int), 3);
  enc2->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  enc2->endEncoding();
  cmdBuf2->commit();

  cmdBuf1->waitUntilCompleted();
  cmdBuf2->waitUntilCompleted();

  wp = wp != nparts - 1 ? wp + 1 : 0;
  wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;

  cmdBuf1 = commands1->commandBuffer();
  enc1 = cmdBuf1->computeCommandEncoder();

  enc1->setComputePipelineState(convol_pso);
  enc1->setBuffer(in1, 0, 0);
  enc1->setBuffer(spec1, 0, 1);
  enc1->setBuffer(spec2, 0, 2);
  enc1->setBytes(&wp, sizeof(int), 3);
  enc1->setBytes(&bins, sizeof(int), 4);
  enc1->setBytes(&nparts, sizeof(int), 5);
  enc1->dispatchThreads(MTL::Size(bsize, 1, 1), MTL::Size(tgSize, 1, 1));

  enc1->setComputePipelineState(c2r_pso);
  enc1->setBuffer(in1, 0, 0);
  enc1->setBuffer(w2[1], 0, 1);
  enc1->setBytes(&bins, sizeof(int), 2);
  enc1->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

  int zero_offs = 0;
  enc1->setComputePipelineState(reorder_pso);
  enc1->setBuffer(out, 0, 0);
  enc1->setBuffer(in1, 0, 1);
  enc1->setBuffer(b, 0, 2);
  enc1->setBytes(&zero_offs, sizeof(int), 3);
  enc1->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc1->setComputePipelineState(fft_pso);
  enc1->setBuffer(out, 0, 0);
  enc1->setBuffer(w[1], 0, 1);
  enc1->setBytes(&bins, sizeof(int), 2);
  for (int n = 1; n < bins; n *= 2) {
    int n2 = n << 1;
    enc1->setBytes(&n2, sizeof(int), 3);
    enc1->setBytes(&zero_offs, sizeof(int), 4);
    enc1->dispatchThreads(MTL::Size(bins >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
  }

  enc1->setComputePipelineState(olap_pso);
  enc1->setBuffer(olap, 0, 0);
  enc1->setBuffer(out, 0, 1);
  enc1->setBytes(&bins, sizeof(int), 2);
  enc1->dispatchThreads(MTL::Size(bins, 1, 1), MTL::Size(tgSize, 1, 1));

  enc1->endEncoding();
  cmdBuf1->commit();
  cmdBuf1->waitUntilCompleted();

  if (!host_mem)
    memcpy(output, olap->contents(), bytes >> 1);

  return 0;
}
}
