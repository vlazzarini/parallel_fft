find_package(CSOUND)
if(NOT CSOUND_INCLUDE_DIRS)
message(status "Csound header files not found; not building Metal opcodes")
else()
if(${CSOUND_VERSION} EQUAL 7)
  set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/metal-cpp")
  add_library(mt_conv SHARED opcode.cpp)
  target_link_libraries(mt_conv mt_fft)
  target_include_directories(mt_conv PRIVATE
    ${CSOUND_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/metal
    ${METAL_CPP_DIR})
  target_link_libraries(mt_conv "-framework Metal" "-framework Foundation")
  set_property(TARGET mt_conv PROPERTY CXX_STANDARD 17)
endif()
message(status "Csound 7 required; not building Metal opcodes")
endif()
