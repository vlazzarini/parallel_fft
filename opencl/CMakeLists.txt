# --- OpenCL targets ---
set(SRCS cl_conv.cpp cl_dconv.cpp cl_fft.cpp)
find_package(OpenCL REQUIRED)

add_library(cl_fft SHARED ${SRCS})
add_executable(test_cfft test_cfft.cpp)
target_link_libraries(test_cfft  cl_fft)
add_executable(test_rfft test_rfft.cpp )
target_link_libraries(test_rfft  cl_fft)

if(APPLE)
target_link_libraries(cl_fft "-framework OpenCL")
else()
target_link_libraries(cl_fft OpenCL)
endif()

set_property(TARGET cl_fft PROPERTY CXX_STANDARD 14)
set_property(TARGET test_cfft PROPERTY CXX_STANDARD 14)
set_property(TARGET test_rfft PROPERTY CXX_STANDARD 14)

add_subdirectory(csound)
