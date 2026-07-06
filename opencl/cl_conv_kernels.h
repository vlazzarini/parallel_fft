////////////////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution kernels
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
////////////////////////////////////////////////////////////////////////////////


const char *pconvcode = R"(  
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
/* atomic add */
inline void AtomicAdd(volatile __global float *source, 
                      const float operand) {
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
/* data reordering */
kernel void reorder(global cmplx *out, global cmplx *in, 
                    global const int *b, int offs) {
  int k = get_global_id(0);
  out += offs;
  out[k] = in[b[k]]; 
  in[b[k]] = 0.f;
}
/* fft stage  */
kernel void fft(global cmplx *s, global const cmplx *w, 
                int N, int n2, int offs) {
  int k, i, m, n;
  cmplx e, o;
  s += offs;
  k = get_global_id(0)*n2;
  m = k/N; 
  n = n2 >> 1;
  k =  k%N + m;
  i = k + n;
  e = s[k];
  o = prod(s[i],w[m*N/n2]);
  s[k] = e + o;
  s[i] = e - o;  
}
/* rfft conversion */
kernel void r2c(global cmplx *c, global const cmplx *w, 
                int N, int offs) {
  int i = get_global_id(0);
  int j = N - i;
  c += offs;
  if(!i%N) {
    c[0] = (cmplx) ((c[0].x + c[0].y)*.5f, (c[0].x - c[0].y)*.5f);
    return;
  }
  cmplx e, o, cj = conjg(c[j]), p;
  e = .5f*(c[i] + cj);
  o = .5f*rot(cj - c[i]);
  p = prod(w[i], o); 
  c[i] = e + p;
  c[j] = conjg(e - p);
}
/* inverse rfft conversion */
kernel void c2r(global cmplx *c, global const cmplx *w, int N) {
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
/* convolution */
kernel void convol(global float *out, global const cmplx *in, 
                   global const cmplx *coef, int rp, int b, 
                   int nparts) {
  /* thread count */
  int k = get_global_id(0); /* bin pos */ 
  int n = k%b;  /* inframe pos   */
  int n2 = n << 1;
  cmplx s;
  rp += k/b;       /*  rp pos */
  /* select correct input buffer */
  in += (rp < nparts ? rp : rp - nparts)*b;  
  /* complex multiplication + sums */
  s = n ? prod(in[n], coef[k]) : 
    (cmplx) (in[0].x*coef[k].x, in[0].y*coef[k].y);
  AtomicAdd(&out[n2], s.x);  
  AtomicAdd(&out[n2+1], s.y);                                
}  
/* sample-by-sample overlap-add operation */
kernel void olap(global float *buf, global const float *in, int parts){
  int n = get_global_id(0);
  buf[n] = (in[n] + buf[parts+n])/parts;
  buf[parts+n] = in[parts+n];
}
)";
