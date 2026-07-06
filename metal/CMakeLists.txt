# --- Metal targets (C++ API) ---
set(MT_SRCS mt_private.mm mt_fft.mm mt_conv.mm mt_dconv.mm)
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/metal-cpp")

add_library(mt_fft SHARED ${MT_SRCS})
target_include_directories(mt_fft PRIVATE ${METAL_CPP_DIR})
target_link_libraries(mt_fft "-framework Metal" "-framework Foundation")

add_executable(test_mt_cfft test_mt_cfft.cpp)
target_include_directories(test_mt_cfft PRIVATE ${METAL_CPP_DIR})
target_link_libraries(test_mt_cfft mt_fft)

add_executable(test_mt_rfft test_mt_rfft.cpp)
target_include_directories(test_mt_rfft PRIVATE ${METAL_CPP_DIR})
target_link_libraries(test_mt_rfft mt_fft)

set_property(TARGET mt_fft PROPERTY CXX_STANDARD 17)
set_property(TARGET test_mt_cfft PROPERTY CXX_STANDARD 17)
set_property(TARGET test_mt_rfft PROPERTY CXX_STANDARD 17)
