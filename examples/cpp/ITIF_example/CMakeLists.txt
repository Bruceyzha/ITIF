add_executable(bert_albert bert_albert.cc)
target_link_libraries(bert_albert PUBLIC -lcublas -lcublasLt -lcudart -lcudart cublasMMWrapper kernel_slice memory_utils )

add_executable(vit_test vit_test.cc)
target_link_libraries(vit_test PUBLIC -lcublas -lcublasLt -lcudart -lcudart new_vit cublasMMWrapper memory_utils -lcudnn -lm)

add_executable(mps mps.cc)
target_link_libraries(mps PUBLIC -lcublas -lcublasLt -lcudart -lcudart new_vit cublasMMWrapper memory_utils -lcudnn -lm)

add_executable(multistream multistream.cc)
target_link_libraries(multistream PUBLIC -lcublas -lcublasLt -lcudart -lcudart new_vit cublasMMWrapper memory_utils -lcudnn -lm)

add_executable(multistream_vit multistream_vit.cc)
target_link_libraries(multistream_vit PUBLIC -lcublas -lcublasLt -lcudart -lcudart new_vit cublasMMWrapper memory_utils -lcudnn -lm)