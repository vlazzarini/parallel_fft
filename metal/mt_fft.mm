/////////////////////////////////////////////////////////////////////
// Metal 1-D Radix-2 FFT implementation (C++ API)
// Port of cl_fft.cpp to Apple Metal
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "mt_fft.h"
#include <vector>

namespace mt_fft {

const char *fft_msl = R"(
typedef float2 cmplx;

kernel void reorder(device cmplx *out [[buffer(0)]],
                    device cmplx *in [[buffer(1)]],
                    constant int *b [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
  out[id] = in[b[id]];
}

kernel void fft(device cmplx *s [[buffer(0)]],
                device cmplx *w [[buffer(1)]],
                constant int &N [[buffer(2)]],
                constant int &n2 [[buffer(3)]],
                constant int &fwd [[buffer(4)]],
                uint id [[thread_position_in_grid]]) {
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
  if (n2 == N && fwd) {
    (s + k)->x = (er + pr) / (float)N; (s + k)->y = (ei + pi) / (float)N;
    (s + i)->x = (er - pr) / (float)N; (s + i)->y = (ei - pi) / (float)N;
  } else {
    (s + k)->x = er + pr; (s + k)->y = ei + pi;
    (s + i)->x = er - pr; (s + i)->y = ei - pi;
  }
}
)";

const char *r2c_msl = R"(
typedef float2 cmplx;

kernel void conv(device cmplx *c [[buffer(0)]],
                 device cmplx *w [[buffer(1)]],
                 constant int &N [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
  if (!id) {
    float c0x = (c + 0)->x, c0y = (c + 0)->y;
    (c + 0)->x = (c0x + c0y) * 0.5f; (c + 0)->y = (c0x - c0y) * 0.5f;
    return;
  }
  int j = N - id;
  float cjx = (c + j)->x, cjy = (c + j)->y;
  float cidx = (c + id)->x, cidy = (c + id)->y;
  float ex = (cidx + cjx) * 0.5f, ey = (cidy - cjy) * 0.5f;
  float ox = (cjy + cidy) * 0.5f, oy = (cjx - cidx) * 0.5f;
  float wr = (w + id)->x, wi = (w + id)->y;
  float px = ox * wr - oy * wi;
  float py = ox * wi + oy * wr;
  (c + id)->x = ex + px; (c + id)->y = ey + py;
  (c + j)->x = ex - px; (c + j)->y = -(ey - py);
}

kernel void iconv(device cmplx *c [[buffer(0)]],
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
)";

Mtcfft::Mtcfft(MTL::Device *dev, int size, bool fwd)
    : N(size), forward(fwd), w(nullptr), b(nullptr), data1(nullptr),
      data2(nullptr), device(dev), commands(nullptr), fft_pso(nullptr),
      reorder_pso(nullptr), mt_err(0) {

  device->retain();

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  commands = device->newCommandQueue();
  if (!commands) { mt_err = -1; pool->release(); return; }

  NS::Error *error = nullptr;
  MTL::Library *lib = device->newLibrary(
      NS::String::string(fft_msl, NS::UTF8StringEncoding), nullptr, &error);
  if (!lib) {
    mt_err = -2;
    pool->release();
    return;
  }

  MTL::Function *fft_fn = lib->newFunction(
      NS::String::string("fft", NS::UTF8StringEncoding));
  MTL::Function *reorder_fn = lib->newFunction(
      NS::String::string("reorder", NS::UTF8StringEncoding));

  fft_pso = device->newComputePipelineState(fft_fn, &error);
  reorder_pso = device->newComputePipelineState(reorder_fn, &error);

  fft_fn->release();
  reorder_fn->release();
  lib->release();
  if (!fft_pso || !reorder_pso) { mt_err = -3; pool->release(); return; }

  size_t bufSize = N * sizeof(float) * 2;
  data1 = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
  data2 = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
  w = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
  b = device->newBuffer(N * sizeof(int), MTL::ResourceStorageModeShared);

  float *wp = (float *)w->contents();
  for (int i = 0; i < N; i++) {
    float sign = forward ? -1.f : 1.f;
    wp[i * 2] = cos(i * 2 * PI / N);
    wp[i * 2 + 1] = sign * sin(i * 2 * PI / N);
  }

  int *bp = (int *)b->contents();
  for (int i = 0; i < N; i++)
    bp[i] = i;
  for (int i = 1, n = N / 2; i < N; i = i << 1, n = n >> 1)
    for (int j = 0; j < i; j++)
      bp[i + j] = bp[j] + n;

  pool->release();
}

Mtcfft::~Mtcfft() {
  w->release();
  b->release();
  data1->release();
  data2->release();
  fft_pso->release();
  reorder_pso->release();
  commands->release();
  device->release();
}

int Mtcfft::fft(MTL::CommandBuffer *cmdBuf) {
  MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();

  enc->setComputePipelineState(reorder_pso);
  enc->setBuffer(data2, 0, 0);
  enc->setBuffer(data1, 0, 1);
  enc->setBuffer(b, 0, 2);
  MTL::Size gridSize = MTL::Size(N, 1, 1);
  NS::UInteger tgSize = reorder_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > N) tgSize = N;
  enc->dispatchThreads(gridSize, MTL::Size(tgSize, 1, 1));

  enc->setComputePipelineState(fft_pso);
  int fwdFlag = forward ? 1 : 0;
  enc->setBuffer(data2, 0, 0);
  enc->setBuffer(w, 0, 1);
  enc->setBytes(&N, sizeof(int), 2);
  enc->setBytes(&fwdFlag, sizeof(int), 4);

  for (int n = 1; n < N; n *= 2) {
    int n2 = n << 1;
    MTL::Size fftGrid = MTL::Size(N >> 1, 1, 1);
    enc->setBytes(&n2, sizeof(int), 3);
    enc->dispatchThreads(fftGrid, MTL::Size(tgSize, 1, 1));
  }

  enc->endEncoding();
  return 0;
}

int Mtcfft::transform(std::complex<float> *c) {
  memcpy(data1->contents(), c, N * sizeof(std::complex<float>));

  MTL::CommandBuffer *cmdBuf = commands->commandBuffer();
  fft(cmdBuf);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  memcpy(c, data2->contents(), N * sizeof(std::complex<float>));
  return 0;
}

Mtrfft::Mtrfft(MTL::Device *dev, int size, bool fwd)
    : w2(nullptr), conv_pso(nullptr), iconv_pso(nullptr),
      Mtcfft(dev, size / 2, fwd) {

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  NS::Error *error = nullptr;
  MTL::Library *lib = device->newLibrary(
      NS::String::string(r2c_msl, NS::UTF8StringEncoding), nullptr, &error);
  if (!lib) { mt_err = -2; pool->release(); return; }

  MTL::Function *conv_fn = lib->newFunction(
      NS::String::string("conv", NS::UTF8StringEncoding));
  MTL::Function *iconv_fn = lib->newFunction(
      NS::String::string("iconv", NS::UTF8StringEncoding));

  conv_pso = device->newComputePipelineState(conv_fn, &error);
  iconv_pso = device->newComputePipelineState(iconv_fn, &error);

  conv_fn->release();
  iconv_fn->release();
  lib->release();
  if (!conv_pso || !iconv_pso) { mt_err = -3; pool->release(); return; }

  w2 = device->newBuffer(N * sizeof(float) * 2, MTL::ResourceStorageModeShared);
  float *wp = (float *)w2->contents();
  for (int i = 0; i < N; i++) {
    float sign = forward ? -1.f : 1.f;
    wp[i * 2] = cos(i * PI / N);
    wp[i * 2 + 1] = sign * sin(i * PI / N);
  }

  pool->release();
}

Mtrfft::~Mtrfft() {
  w2->release();
  conv_pso->release();
  iconv_pso->release();
}

int Mtrfft::transform(std::complex<float> *c, float *r) {
  float *s = reinterpret_cast<float *>(c);
  int N2 = N;
  int fwdFlag = forward ? 1 : 0;
  NS::UInteger tgSize = reorder_pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > N) tgSize = N;

  MTL::CommandBuffer *cmdBuf = commands->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();

  if (forward) {
    if (s != r)
      memcpy(s, r, 2 * N * sizeof(float));

    memcpy(data1->contents(), c, N * sizeof(std::complex<float>));

    enc->setComputePipelineState(reorder_pso);
    enc->setBuffer(data2, 0, 0);
    enc->setBuffer(data1, 0, 1);
    enc->setBuffer(b, 0, 2);
    enc->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->setComputePipelineState(fft_pso);
    enc->setBuffer(data2, 0, 0);
    enc->setBuffer(w, 0, 1);
    enc->setBytes(&N2, sizeof(int), 2);
    enc->setBytes(&fwdFlag, sizeof(int), 4);

    for (int n = 1; n < N; n *= 2) {
      int n2 = n << 1;
      enc->setBytes(&n2, sizeof(int), 3);
      enc->dispatchThreads(MTL::Size(N >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
    }

    enc->setComputePipelineState(conv_pso);
    enc->setBuffer(data2, 0, 0);
    enc->setBuffer(w2, 0, 1);
    enc->setBytes(&N2, sizeof(int), 2);
    enc->dispatchThreads(MTL::Size(N >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    memcpy(c, data2->contents(), N * sizeof(std::complex<float>));
  } else {
    memcpy(data1->contents(), c, N * sizeof(std::complex<float>));

    enc->setComputePipelineState(iconv_pso);
    enc->setBuffer(data1, 0, 0);
    enc->setBuffer(w2, 0, 1);
    enc->setBytes(&N2, sizeof(int), 2);
    enc->dispatchThreads(MTL::Size(N >> 1, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->setComputePipelineState(reorder_pso);
    enc->setBuffer(data2, 0, 0);
    enc->setBuffer(data1, 0, 1);
    enc->setBuffer(b, 0, 2);
    enc->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(tgSize, 1, 1));

    enc->setComputePipelineState(fft_pso);
    enc->setBuffer(data2, 0, 0);
    enc->setBuffer(w, 0, 1);
    enc->setBytes(&N2, sizeof(int), 2);
    enc->setBytes(&fwdFlag, sizeof(int), 4);

    for (int n = 1; n < N; n *= 2) {
      int n2 = n << 1;
      enc->setBytes(&n2, sizeof(int), 3);
      enc->dispatchThreads(MTL::Size(N >> 1, 1, 1), MTL::Size(tgSize, 1, 1));
    }

    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    memcpy(c, data2->contents(), N * sizeof(std::complex<float>));
    if (s != r)
      memcpy(r, s, 2 * N * sizeof(float));
  }
  return 0;
}
}
