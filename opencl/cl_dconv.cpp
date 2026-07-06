/////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution implementation
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "cl_dconv.h"

namespace cl_conv {

const char *dconvcode = R"(
/* atomic add */
inline void AtomicAdd(volatile __global float *source, const float operand) {
     union {
         uint intVal;
         float floatVal;
     } newVal;
     union {
         uint intVal;
         float floatVal;
     } prevVal;
     do {
         prevVal.floatVal = *source;
         newVal.floatVal = prevVal.floatVal + operand;
     } while (atomic_cmpxchg((volatile __global uint *) source, 
       prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
kernel void convol(global float *out, global const float *del, global const 
         float *coefs, int irsize, int rp, int vsize) {
  int t = get_global_id(0);
  float tap;
  if(t >= irsize*vsize) return;
  int n =  t%vsize;  /* sample index */
  int h =  t/vsize;  /* coeff index */
  int end = irsize+vsize;
  rp += n + h; /* read point, oldest -> newest */
  tap = del[rp < end ? rp : rp%end]*coefs[irsize-1-h];  /* single tap */
  AtomicAdd(&out[n], tap);
}
)";

Cldconv::Cldconv(cl_device_id device_id, int cvs, int vsiz,
                 void (*errs)(std::string s, void *d), void *uData)
    : irsize(cvs), vsize(vsiz), wp(0), buff(NULL), coefs(NULL), del(NULL),
      context(NULL), commands(NULL), program(NULL), convol_kernel(NULL), wgs(0),
      err(errs == NULL ? this->msg : errs), userData(uData),
      cl_err(CL_SUCCESS) {

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_err);
  if (!context) {
    err(cl_error_string(cl_err), userData);
    return;
  }
  commands = clCreateCommandQueue(context, device_id, 0, &cl_err);
  if (!commands) {
    err(cl_error_string(cl_err), userData);
    return;
  }
  program = clCreateProgramWithSource(context, 1, (const char **)&dconvcode,
                                      NULL, &cl_err);
  if (!program) {
    err("error creating conv program\n", userData);
    err(cl_error_string(cl_err), userData);
    return;
  }
  cl_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (cl_err) {
    char log[2048];
    size_t llen;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, &llen);
    err("error building conv program\n", userData);
    err(cl_error_string(cl_err), userData);
    err(log, userData);
    return;
  }
  convol_kernel = clCreateKernel(program, "convol", &cl_err);
  clGetKernelWorkGroupInfo(convol_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(wgs), &wgs, NULL);
  if (wgs > vsize * irsize)
    wgs = vsize * irsize;

  buff = clCreateBuffer(context, 0, vsize * sizeof(cl_float), NULL, NULL);
  del = clCreateBuffer(context, CL_MEM_READ_ONLY,
                       (irsize + vsize) * sizeof(cl_float), NULL, NULL);
  coefs = clCreateBuffer(context, CL_MEM_READ_ONLY,
                         (irsize + vsize) * sizeof(cl_float), NULL, NULL);

  clSetKernelArg(convol_kernel, 0, sizeof(cl_mem), &buff);
  clSetKernelArg(convol_kernel, 1, sizeof(cl_mem), &del);
  clSetKernelArg(convol_kernel, 2, sizeof(cl_mem), &coefs);
  clSetKernelArg(convol_kernel, 3, sizeof(cl_int), &irsize);
  clSetKernelArg(convol_kernel, 5, sizeof(cl_int), &vsize);
}

Cldconv::~Cldconv() {
  clReleaseMemObject(del);
  clReleaseMemObject(buff);
  clReleaseMemObject(coefs);
  clReleaseKernel(convol_kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
}

int Cldconv::convolution(float *out, float *in) {
  size_t bytes = vsize * sizeof(cl_float), threads = irsize * vsize;
  char zro = 0;
  if (wp > irsize) {
    int front = wp - irsize;
    bytes = (vsize - front) * sizeof(cl_float);
    clEnqueueWriteBuffer(commands, del, CL_TRUE, wp * sizeof(cl_float), bytes,
                         in, 0, NULL, NULL);
    bytes = front * sizeof(cl_float);
    clEnqueueWriteBuffer(commands, del, CL_TRUE, 0, bytes, &in[vsize - front],
                         0, NULL, NULL);
  } else
    clEnqueueWriteBuffer(commands, del, CL_TRUE, wp * sizeof(cl_float), bytes,
                         in, 0, NULL, NULL);
  clEnqueueFillBuffer(commands, buff, &zro, 1, 0, bytes, 0, NULL, NULL);
  wp = (wp + vsize) % (irsize + vsize);
  clSetKernelArg(convol_kernel, 4, sizeof(cl_int), &wp);
  cl_err = clEnqueueNDRangeKernel(commands, convol_kernel, 1, NULL, &threads,
                                  &wgs, 0, NULL, NULL);
  if (cl_err)
    err(cl_error_string(cl_err), userData);
  clEnqueueReadBuffer(commands, buff, CL_TRUE, 0, bytes, out, 0, NULL, NULL);
  return cl_err;
}

int Cldconv::convolution(float *out, float *in1, float *in2) {
  size_t bytes = vsize * sizeof(cl_float);
  if (wp > irsize) {
    int front = wp - irsize;
    bytes = (vsize - front) * sizeof(cl_float);
    clEnqueueWriteBuffer(commands, coefs, CL_TRUE, wp * sizeof(cl_float), bytes,
                         in2, 0, NULL, NULL);
    bytes = front * sizeof(cl_float);
    clEnqueueWriteBuffer(commands, coefs, CL_TRUE, 0, bytes,
                         &in2[vsize - front], 0, NULL, NULL);
  } else
    clEnqueueWriteBuffer(commands, coefs, CL_TRUE, wp * sizeof(cl_float), bytes,
                         in2, 0, NULL, NULL);
  return convolution(out, in1);
}

int Cldconv::push_ir(float *ir) {
  return clEnqueueWriteBuffer(commands, coefs, CL_TRUE, 0,
                              irsize * sizeof(float), ir, 0, NULL, NULL);
}
}
