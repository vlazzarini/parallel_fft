/*
  opcode.cpp: Metal convolution & FFT opcodes
  Port of opencl/csound/opcode.cpp to Apple Metal

  Copyright (C) 2019-26 Victor Lazzarini
  This file is part of Csound.

  The Csound Library is free software; you can redistribute it
  and/or modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  Csound is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with Csound; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA
*/

#include <mt_fft.h>
#include <mt_conv.h>
#include <mt_dconv.h>
#include <modload.h>
#include <vector>

namespace csnd {
  static inline uint32_t np2(uint32_t n) {
    uint32_t v = 2;
    while (v < n)
       v <<= 1;
    return v;
  }

  void mt_err_msg(std::string s, void *uData) {
    Csound *cs = (Csound *)uData;
    cs->message(s);
  }

  struct MtCfft : Plugin<1, 4> {
    mt_fft::Mtcfft *dft;
    csnd::AuxMem<float> buf;

    int init() {
      Vector<MYFLT> input = inargs.vector_data<MYFLT>(0);
      Vector<MYFLT> output = outargs.vector_data<MYFLT>(0);
      output.init(csound, input.len(), this->insdshead);

      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Array *mtldevices = MTL::CopyAllDevices();
      int nd = mtldevices ? (int)mtldevices->count() : 0;
      if (!nd) {
        pool->drain();
        return csound->init_error("failed to find a Metal device!\n");
      }
      int devidx = (int)inargs[2];
      if (devidx < 0 || devidx >= nd) {
        csound->message("Metal device index out of range, using device 0\n");
        devidx = 0;
      }
      MTL::Device *device = (MTL::Device*)mtldevices->object(devidx);
      const char *name = device->name()->utf8String();
      csound->message("using device: ");
      csound->message(name);
      csound->message("\n");

      dft = new mt_fft::Mtcfft(device, np2(input.len()), inargs[1] ? true : false);
      pool->drain();
      if (dft->get_error())
        return csound->init_error("error initialising Metal FFT");
      buf.allocate(csound, input.len());
      return OK;
    }

    int perf() {
      int i = 0;
      Vector<MYFLT> input = inargs.vector_data<MYFLT>(0);
      Vector<MYFLT> output = outargs.vector_data<MYFLT>(0);
      for (auto s : input)
        buf[i++] = s;
      std::complex<float> *data =
        reinterpret_cast<std::complex<float>*>(buf.data());
      if (dft->transform(data))
        return csound->perf_error("error computing FFT\n", this);
      i = 0;
      for (auto &s : output)
        s = buf[i++];
      return OK;
    }

    int deinit() {
      delete dft;
      return OK;
    }
  };

  struct MtRfft : Plugin<1, 4> {
    mt_fft::Mtrfft *dft;
    csnd::AuxMem<float> buf;

    int init() {
      Vector<MYFLT> input = inargs.vector_data<MYFLT>(0);
      Vector<MYFLT> output = outargs.vector_data<MYFLT>(0);
      output.init(csound, input.len(), this->insdshead);

      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Array *mtldevices = MTL::CopyAllDevices();
      int nd = mtldevices ? (int)mtldevices->count() : 0;
      if (!nd) {
        pool->drain();
        return csound->init_error("failed to find a Metal device!\n");
      }
      int devidx = (int)inargs[2];
      if (devidx < 0 || devidx >= nd) {
        csound->message("Metal device index out of range, using device 0\n");
        devidx = 0;
      }
      MTL::Device *device = (MTL::Device*)mtldevices->object(devidx);
      const char *name = device->name()->utf8String();
      csound->message("using device: ");
      csound->message(name);
      csound->message("\n");

      dft = new mt_fft::Mtrfft(device, np2(input.len()), inargs[1] ? true : false);
      pool->drain();
      if (dft->get_error())
        return csound->init_error("error initialising Metal FFT");
      buf.allocate(csound, input.len());
      return OK;
    }

    int perf() {
      int i = 0;
      Vector<MYFLT> input = inargs.vector_data<MYFLT>(0);
      Vector<MYFLT> output = outargs.vector_data<MYFLT>(0);
      for (auto s : input)
        buf[i++] = s;
      std::complex<float> *data =
        reinterpret_cast<std::complex<float>*>(buf.data());
      if (dft->transform(data))
        return csound->perf_error("error computing FFT\n", this);
      i = 0;
      for (auto &s : output)
        s = buf[i++];
      return OK;
    }

    int deinit() {
      delete dft;
      return OK;
    }
  };

  struct MtConv : Plugin<1, 7> {
    mt_conv::Mtpconv *mtpconv;
    mt_conv::Mtdconv *mtdconv;
    Table ir;
    int parts, cnt;
    bool dconv;
    csnd::AuxMem<float> bufin, bufout;

    int init() {
      int size;
      int err;
      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Array *mtldevices = MTL::CopyAllDevices();
      int nd = mtldevices ? (int)mtldevices->count() : 0;
      if (!nd) {
        pool->drain();
        return csound->init_error("failed to find a Metal device!\n");
      }
      int devidx = (int)inargs[6];
      if (devidx < 0 || devidx >= nd) {
        csound->message("Metal device index out of range, using device 0\n");
        devidx = 0;
      }
      MTL::Device *device = (MTL::Device*)mtldevices->object(devidx);
      const char *name = device->name()->utf8String();
      csound->message("using device: ");
      csound->message(name);
      csound->message("\n");

      ir.init(csound, inargs(1));
      parts = inargs[2];
      size = inargs[5] == 0 ? ir.len() : inargs[5];
      size -= inargs[4];
      MYFLT _0dbfs = csound->_0dbfs();
      dconv = parts == 1 ? true : false;
      if (dconv) {
        int ksmps = insdshead->ksmps;
        mtdconv = new mt_conv::Mtdconv(device, size, ksmps, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtdconv->get_mt_err()) {
          std::vector<float> coefs(size);
          for (int i = 0; i < size; i++)
            coefs[i] = ir[i] * _0dbfs;
          if (!mtdconv->push_ir(coefs.data())) {
            bufout.allocate(csound, ksmps);
            bufin.allocate(csound, ksmps);
            cnt = 0;
            return OK;
          }
          csound->message("error setting impulse response");
        }
        delete mtdconv;
        mtdconv = NULL;
      } else {
        mtpconv = new mt_conv::Mtpconv(device, size, parts, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtpconv->get_mt_err()) {
          std::vector<float> coefs(size);
          for (int i = 0; i < size; i++)
            coefs[i] = ir[i] * _0dbfs;
          if (!mtpconv->push_ir(coefs.data())) {
            bufout.allocate(csound, parts);
            bufin.allocate(csound, parts);
            cnt = 0;
            return OK;
          }
          csound->message("error setting impulse response");
        }
        delete mtpconv;
        mtpconv = NULL;
      }
      return csound->init_error("error initialising Metal convolution");
    }

    int deinit() {
      if (dconv && mtdconv)
       delete mtdconv;
      else if(mtpconv) delete mtpconv;
      return OK;
    }

    int aperf() {
      AudioSig asig(this, inargs(0));
      AudioSig aout(this, outargs(0));

      if (dconv) {
        for (int n = offset; n < nsmps; n++)
          bufin[n] = (float)asig[n];
        if (mtdconv->convolution(bufout.data(), bufin.data()))
          return csound->perf_error("error computing convolution\n", this);
        for (int n = offset; n < nsmps; n++)
          aout[n] = (MYFLT)bufout[n];
      } else {
        for (int n = offset; n < nsmps; n++) {
          bufin[cnt] = (float)asig[n];
          aout[n] = (MYFLT)bufout[cnt];
          if (++cnt == parts) {
            if (mtpconv->convolution(bufout.data(), bufin.data()))
              return csound->perf_error("error computing convolution\n", this);
            mtpconv->synchronise();
            cnt = 0;
          }
        }
      }
      return OK;
    }
  };

  struct MtTVConv : Plugin<1, 7> {
    mt_conv::Mtpconv *mtpconv;
    mt_conv::Mtdconv *mtdconv;
    int parts, cnt;
    bool dconv;
    csnd::AuxMem<float> bufin1, bufin2, bufout;

    int init() {
      int size;
      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Array *mtldevices = MTL::CopyAllDevices();
      int nd = mtldevices ? (int)mtldevices->count() : 0;
      if (!nd) {
        pool->drain();
        return csound->init_error("failed to find a Metal device!\n");
      }
      int devidx = (int)inargs[6];
      if (devidx < 0 || devidx >= nd) {
        csound->message("Metal device index out of range, using device 0\n");
        devidx = 0;
      }
      MTL::Device *device = (MTL::Device*)mtldevices->object(devidx);
      const char *name = device->name()->utf8String();
      csound->message("using device: ");
      csound->message(name);
      csound->message("\n");

      size = inargs[5];
      parts = inargs[4];
      dconv = parts == 1 ? true : false;
      if (dconv) {
        int ksmps = insdshead->ksmps;
        mtdconv = new mt_conv::Mtdconv(device, size, ksmps, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtdconv->get_mt_err()) {
          bufout.allocate(csound, ksmps);
          bufin1.allocate(csound, ksmps);
          bufin2.allocate(csound, ksmps);
          cnt = 0;
          return OK;
        }
        delete mtdconv;
        mtdconv = NULL;
      } else {
        mtpconv = new mt_conv::Mtpconv(device, size, parts, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtpconv->get_mt_err()) {
          cnt = 0;
          bufout.allocate(csound, parts);
          bufin1.allocate(csound, parts);
          bufin2.allocate(csound, parts);
          return OK;
        }
        delete mtpconv;
        mtpconv = NULL;
      }
      return csound->init_error("error initialising Metal convolution");
    }

    int deinit() {
      if (dconv && mtdconv)
       delete mtdconv;
      else if(mtpconv) delete mtpconv;
      return OK;
    }

    int aperf() {
      AudioSig asig1(this, inargs(0));
      AudioSig asig2(this, inargs(1));
      AudioSig aout(this, outargs(0));
      int frz1 = (int)inargs[2], frz2 = (int)inargs[3];
      MYFLT _0dbfs = csound->_0dbfs();

      if (dconv) {
        for (int n = offset; n < nsmps; n++) {
          bufin1[n] = (float)frz1 ? (float)(asig1[n] / _0dbfs) : bufin1[cnt];
          bufin2[n] = (float)frz1 ? (float)(asig2[n] / _0dbfs) : bufin2[cnt];
        }
        if (mtdconv->convolution(bufout.data(), bufin1.data(), bufin2.data()))
          return csound->perf_error("error computing convolution\n", this);
        for (int n = offset; n < nsmps; n++)
          aout[n] = bufout[n] * _0dbfs;
      } else {
        for (int n = offset; n < nsmps; n++) {
          bufin1[cnt] = frz1 ? (float)(asig1[n] / _0dbfs) : bufin1[cnt];
          bufin2[cnt] = frz1 ? (float)(asig2[n] / _0dbfs) : bufin2[cnt];
          aout[n] = bufout[cnt] * _0dbfs;
          if (++cnt == parts) {
            if (mtpconv->convolution(bufout.data(), bufin1.data(),
                                     bufin2.data()))
              return csound->perf_error("error computing convolution\n", this);
            mtpconv->synchronise();
            cnt = 0;
          }
        }
      }
      return OK;
    }
  };

  struct MtTVConvS : Plugin<2, 9> {
    mt_conv::Mtpconv *mtpconvl, *mtpconvr;
    mt_conv::Mtdconv *mtdconvl, *mtdconvr;
    int parts, cnt;
    bool dconv;
    csnd::AuxMem<float> bufin1l, bufin1r, bufin2l, bufin2r,
      bufoutl, bufoutr;

    int init() {
      int size;
      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Array *mtldevices = MTL::CopyAllDevices();
      int nd = mtldevices ? (int)mtldevices->count() : 0;
      if (!nd) {
        pool->drain();
        return csound->init_error("failed to find a Metal device!\n");
      }
      int devidx = (int)inargs[8];
      if (devidx < 0 || devidx >= nd) {
        csound->message("Metal device index out of range, using device 0\n");
        devidx = 0;
      }
      MTL::Device *device = (MTL::Device*)mtldevices->object(devidx);
      const char *name = device->name()->utf8String();
      csound->message("using device: ");
      csound->message(name);
      csound->message("\n");

      size = inargs[7];
      parts = inargs[6];
      dconv = parts == 1 ? true : false;
      if (dconv) {
        int ksmps = insdshead->ksmps;
        mtdconvl = new mt_conv::Mtdconv(device, size, ksmps, mt_err_msg, (void *)csound);
        mtdconvr = new mt_conv::Mtdconv(device, size, ksmps, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtdconvr->get_mt_err() && !mtdconvl->get_mt_err()) {
          bufoutl.allocate(csound, ksmps);
          bufin1l.allocate(csound, ksmps);
          bufin2l.allocate(csound, ksmps);
          bufoutr.allocate(csound, ksmps);
          bufin1r.allocate(csound, ksmps);
          bufin2r.allocate(csound, ksmps);
          cnt = 0;
          return OK;
        }
        delete mtdconvr;
        mtdconvr = NULL;
        delete mtdconvl;
        mtdconvl = NULL;
      } else {
        mtpconvl = new mt_conv::Mtpconv(device, size, parts, mt_err_msg, (void *)csound);
        mtpconvr = new mt_conv::Mtpconv(device, size, parts, mt_err_msg, (void *)csound);
        pool->drain();
        if (!mtpconvl->get_mt_err() && !mtpconvr->get_mt_err()) {
          cnt = 0;
          bufoutl.allocate(csound, parts);
          bufin1l.allocate(csound, parts);
          bufin2l.allocate(csound, parts);
          bufoutr.allocate(csound, parts);
          bufin1r.allocate(csound, parts);
          bufin2r.allocate(csound, parts);
          return OK;
        }
        delete mtpconvr;
        delete mtpconvl;
        mtpconvr = mtpconvl = NULL;
      }
      return csound->init_error("error initialising Metal convolution");
    }

    int deinit() {
      if (dconv && mtdconvl && mtdconvr) {
       delete mtdconvl;
       delete mtdconvr;
      }
      else if(mtpconvr && mtpconvl) {
        delete mtpconvr;
        delete mtpconvl;
      }
      return OK;
    }

    int aperf() {
      AudioSig asig1l(this, inargs(0));
      AudioSig asig2l(this, inargs(1));
      AudioSig asig1r(this, inargs(2));
      AudioSig asig2r(this, inargs(3));
      AudioSig aoutl(this, outargs(0));
      AudioSig aoutr(this, outargs(1));
      int frz1 = (int)inargs[4], frz2 = (int)inargs[5];
      MYFLT _0dbfs = csound->_0dbfs();

      if (dconv) {
        for (int n = offset; n < nsmps; n++) {
          bufin1l[n] = (float)frz1 ? (float)(asig1l[n] / _0dbfs) : bufin1l[cnt];
          bufin1r[n] = (float)frz1 ? (float)(asig1r[n] / _0dbfs) : bufin1r[cnt];
          bufin2r[n] = (float)frz1 ? (float)(asig2r[n] / _0dbfs) : bufin2r[cnt];
          bufin2l[n] = (float)frz1 ? (float)(asig2l[n] / _0dbfs) : bufin2l[cnt];
        }
        if (mtdconvl->convolution(bufoutr.data(), bufin1r.data(), bufin2r.data()) ||
            mtdconvr->convolution(bufoutl.data(), bufin1l.data(), bufin2l.data()))
          return csound->perf_error("error computing convolution\n", this);
        for (int n = offset; n < nsmps; n++){
          aoutl[n] = bufoutl[n] * _0dbfs;
          aoutr[n] = bufoutr[n] * _0dbfs;
        }
      } else {
        for (int n = offset; n < nsmps; n++) {
          bufin1l[cnt] = (float)frz1 ? (float)(asig1l[n] / _0dbfs) : bufin1l[cnt];
          bufin1r[cnt] = (float)frz1 ? (float)(asig1r[n] / _0dbfs) : bufin1r[cnt];
          bufin2r[cnt] = (float)frz1 ? (float)(asig2r[n] / _0dbfs) : bufin2r[cnt];
          bufin2l[cnt] = (float)frz1 ? (float)(asig2l[n] / _0dbfs) : bufin2l[cnt];
          aoutl[n] = bufoutl[cnt] * _0dbfs;
          aoutr[n] = bufoutr[cnt] * _0dbfs;
          if (++cnt == parts) {
            if (mtpconvl->convolution(bufoutr.data(), bufin1r.data(),
                                      bufin2r.data()) ||
                mtpconvr->convolution(bufoutl.data(), bufin1l.data(),
                                      bufin2l.data()))
              return csound->perf_error("error computing convolution\n", this);
            mtpconvl->synchronise();
            mtpconvr->synchronise();
            cnt = 0;
          }
        }
      }
      return OK;
    }
  };

  void on_load(Csound *csound) {
    plugin<MtConv>(csound, "mtconv", "a", "aiiiooo", csnd::thread::ia);
    plugin<MtTVConv>(csound, "mttvconv", "a", "aakkiii", csnd::thread::ia);
    plugin<MtTVConvS>(csound, "mttvconv", "aa", "aaaakkiii", csnd::thread::ia);
    plugin<MtCfft>(csound, "mtfft", "k[]", "k[]iii", csnd::thread::ik);
    plugin<MtRfft>(csound, "mtrfft", "k[]", "k[]iii", csnd::thread::ik);
  }
}
