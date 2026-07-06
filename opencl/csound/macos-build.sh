set -x
c++ -g -O3 -dynamiclib -o libclconv.dylib -std=c++14 -DUSE_DOUBLE -D_FORTIFY_SOURCE=0 opcode.cpp ../cl_conv.cpp ../cl_dconv.cpp -I.. -I/Library/Frameworks/CsoundLib64.framework/Headers -framework OpenCL -Wno-deprecated-declarations -Wno-format-security
