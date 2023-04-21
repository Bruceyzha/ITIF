cmake_minimum_required(VERSION 3.8)
add_library(itif STATIC encoder.cc)
set_property(TARGET itif PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET itif PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(itif PUBLIC PUBLIC -lcublas -lcublasLt -lcudart -lcudart bert_preprocess_kernels cublasMMWrapper 
UnfusedAttentionLayer FfnLayer
add_residual_kernels memory_utils layernorm_kernels changed_layer_norm)