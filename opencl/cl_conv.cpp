////////////////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution implementation
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
////////////////////////////////////////////////////////////////////////////////
#include "cl_conv.h"
#include <vector>
#include <cmath>

namespace cl_conv {

  const double PI = M_PI;

  /** Open CL kernels */
#include "cl_conv_kernels.h"  

  /**  kernel dispatch functions 
       these wrap the kernel arg setting and enqueue calls
       for convenience
 
       Reordering dispatcher
       Used to re-order the data in the fft input array
       via byte-swapped indices stored in the b array
       offs parameter allows access to multiple output
       analysis frames - for overlap-add, partitioned convolution
       NB: on forward fft, real data is taken in and 
       re-interpreted as a complex array (as per rfft convention)
  */
  inline int reorder(cl_mem *out, cl_mem *in, cl_mem *b, int offs,
                     cl_command_queue commands, cl_kernel kern, size_t threads) {
    clSetKernelArg(kern, 3, sizeof(cl_int), &offs);
    clSetKernelArg(kern, 2, sizeof(cl_mem), b);
    clSetKernelArg(kern, 1, sizeof(cl_mem), in);
    clSetKernelArg(kern, 0, sizeof(cl_mem), out);
    return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                  NULL, NULL);
  }

  /** fft dispatcher
      The fft kernel computes a single stage of fft, and is called
      here in a loop log2(bins) times to compute the complete transform.
      Because this is called in an rfft context, bins = N/2.
      The offs parameters is also used here to access multiple frames.
      The transform is computed in place.
  */
  inline int fft(cl_mem *data, cl_mem *w, int bins, int offs,
                 cl_command_queue commands, cl_kernel kern, size_t threads) {
    int cl_err, n2;
    for (int n = 1; n < bins; n *= 2) {
      n2 = n << 1;
      clSetKernelArg(kern, 4, sizeof(cl_int), &offs);
      clSetKernelArg(kern, 3, sizeof(cl_int), &n2);
      clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
      clSetKernelArg(kern, 1, sizeof(cl_mem), w);
      clSetKernelArg(kern, 0, sizeof(cl_mem), data);
      cl_err = clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                      NULL, NULL);
    }
    return cl_err;
  }

  /** r2c dispatcher
      This takes to convert the forward fft result into non-negative
      spectrum with bins complex pairs. The offs parameter allows access to
      multiple input frames.
  */
  inline int real_cmplx(cl_mem *data, cl_mem *w, int bins, int offs,
                        cl_command_queue commands, cl_kernel kern,
                        size_t threads) {
    clSetKernelArg(kern, 3, sizeof(cl_int), &offs);
    clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
    clSetKernelArg(kern, 1, sizeof(cl_mem), w);
    clSetKernelArg(kern, 0, sizeof(cl_mem), data);
    return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                  NULL, NULL);
  }

  /** c2r dispatcher
      This is used to prepare the result for the inverse transform
      (reorder + fft). Note that since there is only one output
      frame at a time, there is no need for an offs parameter
  */
  inline int cmplx_real(cl_mem *data, cl_mem *w, int bins,
                        cl_command_queue commands, cl_kernel kern,
                        size_t threads) {
    clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
    clSetKernelArg(kern, 1, sizeof(cl_mem), w);
    clSetKernelArg(kern, 0, sizeof(cl_mem), data);
    return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                  NULL, NULL);
  }

  /** partitioned convolution dispatcher
      This takes two inputs (in and coefs) and computes a partition of 
      the spectrum of the convolution of these two. The input array
      is a circular buffer containing input frames for each partition.
      The coeffs is also a circular buffer, with frames stored in 
      reverse order. The kernel operation consists of the spectral
      product of all frames in the two buffers into an output buffer
      and the total sum of all frames in this (atomic op required). 
  */
  inline int convol(cl_mem *out, cl_mem *in, cl_mem *coefs, int wp, int bins,
                    int nparts, cl_command_queue commands, cl_kernel kern,
                    size_t threads) {
    clSetKernelArg(kern, 5, sizeof(cl_int), &nparts);
    clSetKernelArg(kern, 4, sizeof(cl_int), &bins);
    clSetKernelArg(kern, 3, sizeof(cl_int), &wp);
    clSetKernelArg(kern, 2, sizeof(cl_mem), coefs);
    clSetKernelArg(kern, 1, sizeof(cl_mem), in);
    clSetKernelArg(kern, 0, sizeof(cl_mem), out);
    return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                  NULL, NULL);
  }

  /** ola dispatcher
      Following the inverse fft, we access the transform buffer, now 
      reinterpreted as a real-valued array (as per rfft convention), and 
      overlap-add into the output (saving the second half of the array).
      The output size is parts (floats).
  */ 
  inline int ola(cl_mem *out, cl_mem *in, int parts, cl_command_queue commands,
                 cl_kernel kern, size_t threads) {
    clSetKernelArg(kern, 2, sizeof(cl_int), &parts);
    clSetKernelArg(kern, 1, sizeof(cl_mem), in);
    clSetKernelArg(kern, 0, sizeof(cl_mem), out);
    return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                  NULL, NULL);
  }

  /** Constructor
      takes care of all setup for OpenCL kernels and buffers.
  */
  Clpconv::Clpconv(cl_device_id device_id, int cvs, int pts,
                   void (*errs)(std::string s, void *d), void *uData, void *inp1,
                   void *inp2, void *outp)
    : N(pts << 1), bins(pts), bsize((cvs / pts) * bins), nparts(cvs / pts),
      wp(0), wp2(nparts - 1), w{NULL, NULL}, w2{NULL, NULL}, b(NULL), in1(NULL),
      in2(NULL), out(NULL), spec1(NULL), spec2(NULL), olap(NULL), context(NULL),
      commands1(NULL), commands2(NULL), program(NULL), fft_kernel1(NULL),
      reorder_kernel1(NULL), fft_kernel2(NULL), reorder_kernel2(NULL),
      r2c_kernel1(NULL), r2c_kernel2(NULL), c2r_kernel(NULL),
      convol_kernel(NULL), olap_kernel(NULL),
      err(errs == NULL ? this->msg : errs), userData(uData), cl_err(CL_SUCCESS),
      host_mem(((uintptr_t)inp1 & (uintptr_t)inp2 & (uintptr_t)out) ? 1 : 0) {

        /* Create OpenCL compute context object on device device_id */
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_err);
        if (!context) {
          err(cl_error_string(cl_err), userData);
          return;
        }

        /* Create OpenCL command queues, we need
           two command queues for task parallelism in the forward
           part of the process (dealing with two inputs)
        */
        commands1 = clCreateCommandQueue(context, device_id, 0, &cl_err);
        if (!commands1) {
          err(cl_error_string(cl_err), userData);
          return;
        }
        commands2 = clCreateCommandQueue(context, device_id, 0, &cl_err);
        if (!commands2) {
          err(cl_error_string(cl_err), userData);
          return;
        }

        /* Create the OpenCL program object  */
        program = clCreateProgramWithSource(context, 1, (const char **)&pconvcode,
                                            NULL, &cl_err);
        if (!program) {
          err("error creating opencl program\n", userData);
          err(cl_error_string(cl_err), userData);
          return;
        }

        /* Build the OpenCL program with jit compiler */
        cl_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (cl_err) {
          char log[2048];
          size_t llen;
          clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log),
                                log, &llen);
          err("error building opencl program\n", userData);
          err(cl_error_string(cl_err), userData);
          err(log, userData);
          return;
        }

        /* Now we create the OpenCL kernel objects for the dispatchers,
           separate kernel objects are needed for each command queue 
        */
        reorder_kernel1 = clCreateKernel(program, "reorder", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        fft_kernel1 = clCreateKernel(program, "fft", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        r2c_kernel1 = clCreateKernel(program, "r2c", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        reorder_kernel2 = clCreateKernel(program, "reorder", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        fft_kernel2 = clCreateKernel(program, "fft", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        r2c_kernel2 = clCreateKernel(program, "r2c", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        c2r_kernel = clCreateKernel(program, "c2r", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        convol_kernel = clCreateKernel(program, "convol", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);
        olap_kernel = clCreateKernel(program, "olap", &cl_err);
        if (cl_err != 0)
          err(cl_error_string(cl_err), userData);

        /* OpenCL buffers for kernels
           The first three are used to pass data to/from device
           and may be optionally allocated by the host
        */
        in1 = clCreateBuffer(context, in1 ? CL_MEM_USE_HOST_PTR : 0,
                             bins * sizeof(cl_float2), inp1, &cl_err);
        in2 = clCreateBuffer(context, in2 ? CL_MEM_USE_HOST_PTR : 0,
                             bins * sizeof(cl_float2), inp2, &cl_err);
        olap = clCreateBuffer(context, out ? CL_MEM_USE_HOST_PTR : 0,
                              bins * sizeof(cl_float2), outp, &cl_err);

        /* used to hold the inverse fft results */
        out = clCreateBuffer(context, 0, bins * sizeof(cl_float2), NULL, &cl_err);

        /* two input buffers - holding spectral frames for all partitions */
        spec1 = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                               NULL, &cl_err);
        spec2 = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                               NULL, &cl_err);

        /* four arrays for pre-computed twiddle table constants */
        w[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                              NULL, &cl_err);
        w[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                              NULL, &cl_err);
        w2[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                               NULL, &cl_err);
        w2[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                               NULL, &cl_err);
        
        /* one array for bit-reverse indices for reorder operation */
        b = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_int), NULL,
                           NULL);
        
        /* Compute twiddle tables */
        std::vector<std::complex<float>> wd(bins);
        for (int i = 0; i < bins; i++) {
          wd[i].real(cos(i * 2 * PI / bins));
          wd[i].imag(-sin(i * 2 * PI / bins));
        }
        clEnqueueWriteBuffer(commands1, w[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)wd.data(), 0, NULL, NULL);
        for (int i = 0; i < bins; i++) {
          wd[i].real(cos(i * 2 * PI / bins));
          wd[i].imag(sin(i * 2 * PI / bins));
        }
        clEnqueueWriteBuffer(commands1, w[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)wd.data(), 0, NULL, NULL);
        for (int i = 0; i < bins; i++) {
          wd[i].real(cos(i * PI / bins));
          wd[i].imag(-sin(i * PI / bins));
        }
        clEnqueueWriteBuffer(commands1, w2[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)wd.data(), 0, NULL, NULL);
        for (int i = 0; i < bins; i++) {
          wd[i].real(cos(i * PI / bins));
          wd[i].imag(sin(i * PI / bins));
        }
        clEnqueueWriteBuffer(commands1, w2[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)wd.data(), 0, NULL, NULL);

        /* compute bit-reversed indices */
        std::vector<int> bp(bins);
        for (int i = 0; i < bins; i++)
          bp[i] = i;
        for (int i = 1, n = bins / 2; i<bins; i = i << 1, n = n>> 1)
          for (int j = 0; j < i; j++)
            bp[i + j] = bp[j] + n;
        clEnqueueWriteBuffer(commands1, b, CL_TRUE, 0, sizeof(cl_int) * bins,
                             (const void *)bp.data(), 0, NULL, NULL);

        /** Initialise buffers
            OpenCL does not seem to have a memset/fill 
            so we just copy zeros in 
        */
        std::vector<float> zeros(bsize * 2, 0);
        clEnqueueWriteBuffer(commands1, olap, CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)zeros.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(commands1, spec1, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                             (const void *)zeros.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(commands1, spec2, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                             (const void *)zeros.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(commands1, in1, CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)zeros.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(commands1, in2, CL_TRUE, 0, sizeof(cl_float2) * bins,
                             (const void *)zeros.data(), 0, NULL, NULL);
        /** 
            done; note that 
            N - transform size (we do not use this anywhere, but keep for sanity)
            bins - number of bins and partition size
            nparts - number of partitions (computed from convolution size)
        */
      }

  Clpconv::~Clpconv() {
    /* free all OpenCL memory */
    clReleaseMemObject(w2[0]);
    clReleaseMemObject(w2[1]);
    clReleaseMemObject(w[0]);
    clReleaseMemObject(w[1]);
    clReleaseMemObject(b);
    clReleaseMemObject(out);
    clReleaseMemObject(in2);
    clReleaseMemObject(in1);
    clReleaseMemObject(olap);
    clReleaseMemObject(spec1);
    clReleaseMemObject(spec2);
    clReleaseKernel(fft_kernel2);
    clReleaseKernel(reorder_kernel2);
    clReleaseKernel(r2c_kernel2);
    clReleaseKernel(fft_kernel1);
    clReleaseKernel(reorder_kernel1);
    clReleaseKernel(r2c_kernel1);
    clReleaseKernel(c2r_kernel);
    clReleaseKernel(olap_kernel);
    clReleaseKernel(convol_kernel);
    clReleaseCommandQueue(commands1);
    clReleaseCommandQueue(commands2);
    clReleaseContext(context);
  }
  /** This method does the forward transforms to fill the
      coeffs buffer with nparts frames. 
      The input should have nparts * bins samples, otherwise
      there'll be a crash.
  */
  int Clpconv::push_ir(float *ir) {
    size_t bytes = sizeof(cl_float2) * bins;
    size_t threads;
    int n2;
    /* for each partition of the ir */
    for (int i = 0; i < nparts; i++) {
      /* copy into 1/2 of the in2 buffer */
      if (!host_mem)
        cl_err = clEnqueueWriteBuffer(commands1, in2, CL_TRUE, 0, bytes >> 1,
                                      &ir[i * bins], 0, NULL, NULL);
      if (cl_err != CL_SUCCESS)
        return cl_err;

      /* reorder data into respective frame of spec2 */
      cl_err =
        reorder(&spec2, &in2, &b, wp2 * bins, commands1, reorder_kernel1, bins);
      if (cl_err != CL_SUCCESS)
        return cl_err;

      /* take the fft */
      cl_err =
        fft(&spec2, &w[0], bins, wp2 * bins, commands1, fft_kernel1, bins >> 1);
      if (cl_err != CL_SUCCESS)
        return cl_err;

      /* convert the data for rfft */
      cl_err = real_cmplx(&spec2, &w2[0], bins, wp2 * bins, commands1,
                          r2c_kernel1, bins >> 1);
      if (cl_err != CL_SUCCESS)
        return cl_err;

      /* decrement the circular buffer position */
      wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;
    }
    return CL_SUCCESS;
  }

  /** partitioned convolution,
      one input and coefs from table
  */
  int Clpconv::convolution(float *output, float *input) {
    int n2;
    size_t bytes = sizeof(cl_float2) * bins;
    char zro = 0;
    /* read real-valued data into 1/2 in1 buffer */
    if (!host_mem)
      cl_err = clEnqueueWriteBuffer(commands1, in1, 0, 0, bytes >> 1, input, 0,
                                    NULL, NULL);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Take in1 data, reinterpret it as complex, reorder and fill 
       the respective spec1 buffer partition  */
    cl_err =
      reorder(&spec1, &in1, &b, wp * bins, commands1, reorder_kernel1, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute fft in-place in a spec1 partition  */
    cl_err =
      fft(&spec1, &w[0], bins, wp * bins, commands1, fft_kernel1, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute the rfft conversion in-place in a spec1 partition */
    cl_err = real_cmplx(&spec1, &w2[0], bins, wp * bins, commands1, r2c_kernel1,
                        bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* increment the circular buffer position */
    wp = wp != nparts - 1 ? wp + 1 : 0;

    /* compute a partition of the spectrum of the convolution of
       input and ir */
    cl_err = convol(&in1, &spec1, &spec2, wp, bins, nparts, commands1,
                    convol_kernel, bsize);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* compute the inverse rfft conversion of the output spectrum */
    cl_err = cmplx_real(&in1, &w2[1], bins, commands1, c2r_kernel, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* reorder the output spectrum for inverse fft */
    cl_err = reorder(&out, &in1, &b, 0, commands1, reorder_kernel1, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* compute inverse fft */
    cl_err = fft(&out, &w[1], bins, 0, commands1, fft_kernel1, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* overlap-add the fft output into the olap buffer */
    cl_err = ola(&olap, &out, bins, commands1, olap_kernel, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* copy 1/2 olap buffer to the output */
    if (!host_mem)
      cl_err = clEnqueueReadBuffer(commands1, olap, CL_TRUE, 0, bytes >> 1,
                                   output, 0, NULL, NULL);
    return cl_err;
  }

  int Clpconv::convolution(float *output, float *input1, float *input2) {
    size_t bytes = sizeof(cl_float2) * bins;

    /* read real-valued data into 1/2 in1 buffer */
    if (!host_mem)
      cl_err = clEnqueueWriteBuffer(commands1, in1, 0, 0, bytes >> 1, input1, 0,
                                    NULL, NULL); 
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* read real-valued data into 1/2 in2 buffer */
    if (!host_mem)
      cl_err = clEnqueueWriteBuffer(commands2, in2, 0, 0, bytes >> 1, input2, 0,
                                    NULL, NULL);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Take in1 data, reinterpret it as complex, reorder and fill 
       the respective spec1 buffer partition  */
    cl_err =
      reorder(&spec1, &in1, &b, wp * bins, commands1, reorder_kernel1, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Take in2 data, reinterpret it as complex, reorder and fill 
       the respective spec2 buffer partition  */
    cl_err =
      reorder(&spec2, &in2, &b, wp2 * bins, commands2, reorder_kernel2, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute fft in-place in a spec1 partition  */
    cl_err =
      fft(&spec1, &w[0], bins, wp * bins, commands1, fft_kernel1, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute fft in-place in a spec2 partition  */
    cl_err =
      fft(&spec2, &w[0], bins, wp2 * bins, commands2, fft_kernel2, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute the rfft conversion in-place in a spec1 partition */
    cl_err = real_cmplx(&spec1, &w2[0], bins, wp * bins, commands1, r2c_kernel1,
                        bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* Compute the rfft conversion in-place in a spec2 partition */
    cl_err = real_cmplx(&spec2, &w2[0], bins, wp2 * bins, commands2, r2c_kernel2,
                        bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* increment spec1 circular buffer position */
    wp = wp != nparts - 1 ? wp + 1 : 0;
    
    /* decrement spec2 circular buffer position */
    wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;

    /* synchronise command queue 2 - wait until complete */
    clFinish(commands2);

    /* compute a partition of the spectrum of the convolution of
       input1 and input2 */
    cl_err = convol(&in1, &spec1, &spec2, wp, bins, nparts, commands1,
                    convol_kernel, bsize);
    if (cl_err != CL_SUCCESS)
      return cl_err;

    /* inverse operations - c2r, reorder, fft, and ola as above */
    cl_err = cmplx_real(&in1, &w2[1], bins, commands1, c2r_kernel, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = reorder(&out, &in1, &b, 0, commands1, reorder_kernel1, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = fft(&out, &w[1], bins, 0, commands1, fft_kernel1, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = ola(&olap, &out, bins, commands1, olap_kernel, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    if (!host_mem)
      cl_err = clEnqueueReadBuffer(commands1, olap, 0, 0, bytes >> 1,
                                   output, 0, NULL, NULL);
    return cl_err;
  }
}
