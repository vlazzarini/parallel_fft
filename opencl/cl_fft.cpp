/////////////////////////////////////////////////////////////////////
// OpenCL 1-D Radix-2 FFT implentation
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "cl_fft.h"
#include <vector>

namespace cl_fft {

const char *fft_code = R"(
/* complex type */
typedef float2 cmplx;
/* complex product */
inline cmplx prod(cmplx a, cmplx b){
     return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); 
}
/* reorder kernel */
kernel void reorder(global cmplx *out, global cmplx *in, global const int *b) {
   int k = get_global_id(0);
   out[k] = in[b[k]];     
}
/* fft kernel */
kernel void fft(global cmplx *s, global const cmplx *w, int N, int n2, int fwd) {
 int k, i, m, n;
 cmplx e, o;
 k = get_global_id(0)*n2;
 m = k/N; 
 n = n2 >> 1;
 k =  k%N + m;
 i = k + n;
 e = s[k];
 o = prod(s[i],w[m*N/n2]);
 s[k] = n2 == N && fwd ? (e + o)/N :  e + o;
 s[i] = n2 == N && fwd ? (e - o)/N :  e - o; 
}
)";

Clcfft::Clcfft(cl_device_id device_id, int size, bool fwd)
    : N(size), forward(fwd), w(NULL), b(NULL), data1(NULL), data2(NULL),
      context(NULL), commands(NULL), program(NULL), fft_kernel(NULL),
      reorder_kernel(NULL), wgs(size / 4), cl_err(0) {

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_err);
  if (context) {
    commands = clCreateCommandQueue(context, device_id, 0, &cl_err);
    if (commands) {
      program = clCreateProgramWithSource(context, 1, (const char **)&fft_code,
                                          NULL, &cl_err);
      if (program) {
        cl_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (cl_err) {
          clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                sizeof(log), log, &llen);
          clReleaseCommandQueue(commands);
          clReleaseContext(context);
          return;
        }
        fft_kernel = clCreateKernel(program, "fft", &cl_err);

        clGetKernelWorkGroupInfo(fft_kernel, device_id,
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs,
                                 NULL);
        if (wgs > N / 2)
          wgs = N / 2;

        reorder_kernel = clCreateKernel(program, "reorder", &cl_err);
        clGetKernelWorkGroupInfo(reorder_kernel, device_id,
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(rwgs), &rwgs,
                                 NULL);
        if (rwgs > N)
          rwgs = N;

        data1 = clCreateBuffer(context, 0, N * sizeof(cl_float2), NULL, NULL);
        data2 = clCreateBuffer(context, 0, N * sizeof(cl_float2), NULL, NULL);
        w = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(cl_float2),
                           NULL, NULL);
        b = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(cl_int), NULL,
                           NULL);
        /* twiddle */
        std::vector<std::complex<float>> wp(N);
        for (int i = 0; i < N; i++) {
          float sign = forward ? -1.f : 1.f;
          wp[i].real(cos(i * 2 * PI / N));
          wp[i].imag(sign * sin(i * 2 * PI / N));
        }
        clEnqueueWriteBuffer(commands, w, CL_TRUE, 0, sizeof(cl_float2) * N,
                             (const void *)wp.data(), 0, NULL, NULL);

        /* bit-reversed indices */
        std::vector<int> bp(N);
        for (int i = 0; i < N; i++)
          bp[i] = i;
        for (int i = 1, n = N / 2; i<N; i = i << 1, n = n>> 1)
          for (int j = 0; j < i; j++)
            bp[i + j] = bp[j] + n;

        clEnqueueWriteBuffer(commands, b, CL_TRUE, 0, sizeof(cl_int) * N,
                             (const void *)bp.data(), 0, NULL, NULL);

        int fwd = forward ? 1 : 0;
        clSetKernelArg(reorder_kernel, 0, sizeof(cl_mem), &data2);
        clSetKernelArg(reorder_kernel, 1, sizeof(cl_mem), &data1);
        clSetKernelArg(reorder_kernel, 2, sizeof(cl_mem), &b);
        clSetKernelArg(fft_kernel, 0, sizeof(cl_mem), &data2);
        clSetKernelArg(fft_kernel, 1, sizeof(cl_mem), &w);
        clSetKernelArg(fft_kernel, 2, sizeof(cl_int), &N);
        clSetKernelArg(fft_kernel, 4, sizeof(cl_int), &fwd);

        clReleaseProgram(program);
        return;
      }
      // program not created
      clReleaseCommandQueue(commands);
    }
    // commands not created
    clReleaseContext(context);
  }
  // context not created
}

Clcfft::~Clcfft() {
  clReleaseMemObject(w);
  clReleaseMemObject(b);
  clReleaseMemObject(data1);
  clReleaseMemObject(data2);
  clReleaseKernel(fft_kernel);
  clReleaseKernel(reorder_kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
}

int Clcfft::fft() {
  int err;
  size_t threads = N;
  err = clEnqueueNDRangeKernel(commands, reorder_kernel, 1, NULL, &threads,
                               &rwgs, 0, NULL, NULL);
  for (int n = 1; n < N; n *= 2) {
    int n2 = n << 1;
    threads = N >> 1;
    clSetKernelArg(fft_kernel, 3, sizeof(cl_int), &n2);
    err = clEnqueueNDRangeKernel(commands, fft_kernel, 1, NULL, &threads, &wgs,
                                 0, NULL, NULL);
  }
  return err;
}

int Clcfft::transform(std::complex<float> *c) {
  int err;
  clEnqueueWriteBuffer(commands, data1, CL_TRUE, 0, sizeof(cl_float2) * N, c, 0,
                       NULL, NULL);
  err = fft();
  clEnqueueReadBuffer(commands, data2, CL_TRUE, 0, sizeof(cl_float2) * N, c, 0,
                      NULL, NULL);
  return err;
}
const char *r2c_code = R"(
/* complex type */
typedef float2 cmplx;
/* complex product */
inline cmplx prod(cmplx a, cmplx b){
     return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); 
}
/* complex conj */
inline cmplx conjg(cmplx a) {
    return (cmplx) (a.x, - a.y);
}
/* rotation by pi */
inline cmplx rot(cmplx a) {
   return (cmplx) (-a.y, a.x);
}  
/* conversion kernels */
kernel void conv(global cmplx *c, global const cmplx *w, int N) {
  int i = get_global_id(0);
  if(!i) {
   c[0] = (cmplx) ((c[0].x + c[0].y)*.5f, (c[0].x - c[0].y)*.5f);
   return;
  }
  int j = N - i;
  cmplx e, o, cj = conjg(c[j]), p;
  e = .5f*(c[i] + cj);
  o = .5f*rot(cj - c[i]);
  p = prod(w[i], o); 
  c[i] = e + p;
  c[j] = conjg(e - p);
}
kernel void iconv(global cmplx *c, global const cmplx *w, int N) {
  int i = get_global_id(0);
  if(!i) {
   c[0] = (cmplx) ((c[0].x + c[0].y), (c[0].x - c[0].y));
   return; 
  }
  int j = N - i;
  cmplx e, o, cj = conjg(c[j]), p;
  e = .5f*(c[i] + cj);
  o = .5f*rot(c[i] - cj);
  p = prod(w[i], o);
  c[i] = e + p;
  c[j] = conjg(e - p); 
}
)";

Clrfft::Clrfft(cl_device_id device_id, int size, bool fwd)
    : w2(NULL), conv_kernel(NULL), iconv_kernel(NULL), cwgs(size / 8),
      iwgs(size / 8), Clcfft(device_id, size / 2, fwd) {
  int err;

  program = clCreateProgramWithSource(context, 1, (const char **)&r2c_code,
                                      NULL, &cl_err);
  if (program) {
    cl_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (cl_err) {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                            sizeof(log), log, &llen);
      clReleaseProgram(program);
      clReleaseCommandQueue(commands);
      clReleaseContext(context);
      return;
    }
  }

  conv_kernel = clCreateKernel(program, "conv", &err);
  iconv_kernel = clCreateKernel(program, "iconv", &err);
  w2 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(cl_float2), NULL,
                      NULL);

  /* twiddle */
  std::vector<std::complex<float>> wp(N);
  for (int i = 0; i < N; i++) {
    float sign = forward ? -1.f : 1.f;
    wp[i].real(cos(i * PI / N));
    wp[i].imag(sign * sin(i * PI / N));
  }

  clEnqueueWriteBuffer(commands, w2, CL_TRUE, 0, sizeof(cl_float2) * N,
                       (const void *)wp.data(), 0, NULL, NULL);

  clGetKernelWorkGroupInfo(conv_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(wgs), &cwgs, NULL);
  if (cwgs > N / 2)
    cwgs = N / 2;
  clGetKernelWorkGroupInfo(iconv_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(wgs), &iwgs, NULL);
  if (iwgs > N / 2)
    iwgs = N / 2;

  clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &data2);
  clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &w2);
  clSetKernelArg(conv_kernel, 2, sizeof(cl_int), &N);
  clSetKernelArg(iconv_kernel, 0, sizeof(cl_mem), &data1);
  clSetKernelArg(iconv_kernel, 1, sizeof(cl_mem), &w2);
  clSetKernelArg(iconv_kernel, 2, sizeof(cl_int), &N);
  clReleaseProgram(program);
}

Clrfft::~Clrfft() {
  clReleaseMemObject(w2);
  clReleaseKernel(iconv_kernel);
  clReleaseKernel(conv_kernel);
}

int Clrfft::transform(std::complex<float> *c, float *r) {
  int err;
  float zro, nyq;
  float *s = reinterpret_cast<float *>(c);

  if (forward) {
    if (s != r)
      std::copy(r, r + 2 * N, s);
    clEnqueueWriteBuffer(commands, data1, CL_TRUE, 0, sizeof(cl_float2) * N, c,
                         0, NULL, NULL);
    fft();
    size_t threads = N >> 1;
    err = clEnqueueNDRangeKernel(commands, conv_kernel, 1, NULL, &threads,
                                 &cwgs, 0, NULL, NULL);
    clEnqueueReadBuffer(commands, data2, CL_TRUE, 0, sizeof(cl_float2) * N, c,
                        0, NULL, NULL);
  } else {
    clEnqueueWriteBuffer(commands, data1, CL_TRUE, 0, sizeof(cl_float2) * N, c,
                         0, NULL, NULL);
    size_t threads = N >> 1;
    err = clEnqueueNDRangeKernel(commands, iconv_kernel, 1, NULL, &threads,
                                 &iwgs, 0, NULL, NULL);
    fft();
    clEnqueueReadBuffer(commands, data2, CL_TRUE, 0, sizeof(cl_float2) * N, c,
                        0, NULL, NULL);
    if (s != r)
      std::copy(s, s + 2 * N, r);
  }
  return err;
}

const char *cl_error_string(int err) {
  switch (err) {
  case CL_SUCCESS:
    return "Success!";
  case CL_DEVICE_NOT_FOUND:
    return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:
    return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:
    return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES:
    return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY:
    return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "Profiling information not available";
  case CL_MEM_COPY_OVERLAP:
    return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE:
    return "Program build failure";
  case CL_MAP_FAILURE:
    return "Map failure";
  case CL_INVALID_VALUE:
    return "Invalid value";
  case CL_INVALID_DEVICE_TYPE:
    return "Invalid device type";
  case CL_INVALID_PLATFORM:
    return "Invalid platform";
  case CL_INVALID_DEVICE:
    return "Invalid device";
  case CL_INVALID_CONTEXT:
    return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:
    return "Invalid command queue";
  case CL_INVALID_HOST_PTR:
    return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:
    return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE:
    return "Invalid image size";
  case CL_INVALID_SAMPLER:
    return "Invalid sampler";
  case CL_INVALID_BINARY:
    return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:
    return "Invalid build options";
  case CL_INVALID_PROGRAM:
    return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "Invalid program executable";
  case CL_INVALID_KERNEL_NAME:
    return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:
    return "Invalid kernel definition";
  case CL_INVALID_KERNEL:
    return "Invalid kernel";
  case CL_INVALID_ARG_INDEX:
    return "Invalid argument index";
  case CL_INVALID_ARG_VALUE:
    return "Invalid argument value";
  case CL_INVALID_ARG_SIZE:
    return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:
    return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:
    return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:
    return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "Invalid event wait list";
  case CL_INVALID_EVENT:
    return "Invalid event";
  case CL_INVALID_OPERATION:
    return "Invalid operation";
  case CL_INVALID_GL_OBJECT:
    return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:
    return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:
    return "Invalid mip-map level";
  default:
    return "Unknown error";
  }
}
}
