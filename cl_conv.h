/////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution class
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __CL_CONV_H__
#define __CL_CONV_H__
#include <complex>
#include <iostream>
#include <string>

#ifdef __MACH__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

namespace cl_conv {

inline const char *cl_string(int err) {
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

class Clpconv {
  int N, bins;
  int bsize, nparts, wp, wp2;
  cl_mem w[2], w2[2], b;
  cl_mem in1, in2, out;
  cl_mem spec1, spec2, olap;
  cl_context context;
  cl_command_queue commands1, commands2;
  cl_program program;
  cl_kernel fft_kernel1, reorder_kernel1;
  cl_kernel fft_kernel2, reorder_kernel2;
  cl_kernel r2c_kernel1, r2c_kernel2, c2r_kernel;
  cl_kernel convol_kernel, olap_kernel;
  void (*err)(std::string s, void *uData);
  void *userData;
  int cl_err;
  int host_mem;

  static void msg(std::string str, void *userData) {
    if (userData == NULL)
      std::cout << str << std::endl;
  }

public:
  /** Constructor \n
      device_id - OpenCL device ID \n
      cvs - impulse response size \n
      pts - partition size \n
      errs - error message callback \n
      uData - callback user data \n
      in1, in2, out - host-allocated arrays (optional)
  */
  Clpconv(cl_device_id device_id, int cvs, int pts,
          void (*errs)(std::string s, void *d) = NULL, void *uData = NULL,
          void *in1 = NULL, void *in2 = NULL, void *out = 0);
  ~Clpconv();

  /** returns an error string relative to error code err */
  const char *cl_error_string(int err) { return cl_string(err); }

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

  /** get a recorded error code, CL_SUCCESS if no error was recorded
   */
  int get_cl_err() { return cl_err; }

  /** this needs to be called after convolution to
      synchronise threads */
  void synchronise() { clFinish(commands1); }
};
}
#endif
