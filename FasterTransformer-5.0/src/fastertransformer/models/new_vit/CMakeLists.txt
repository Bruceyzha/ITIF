cmake_minimum_required(VERSION 3.8)

add_library(new_vit STATIC new_vit.cc)
set_property(TARGET new_vit PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET new_vit PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(new_vit PUBLIC -lcudart -lcublasLt -lcublas cublasMMWrapper 
                      layernorm_kernels kernel_slice changed_layer_norm
                      add_residual_kernels activation_kernels vit_kernels bert_preprocess_kernels)
