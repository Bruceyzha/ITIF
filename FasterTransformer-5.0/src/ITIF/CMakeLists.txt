cmake_minimum_required(VERSION 3.8)

add_library(ITIF STATIC ITIF.cc)
set_property(TARGET ITIF PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ITIF PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ITIF PUBLIC -lcudart -lcublasLt -lcublas cublasMMWrapper 
                      layernorm_kernels itif )
