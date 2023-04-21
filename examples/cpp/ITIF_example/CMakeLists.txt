add_executable(bert_albert bert_albert.cc)
target_link_libraries(bert_albert PUBLIC -lcublas -lcublasLt -lcudart -lcudart cublasMMWrapper kernel_slice memory_utils )

add_executable(vit_test vit_test.cc)
target_link_libraries(vit_test PUBLIC -lcublas -lcublasLt -lcudart -lcudart new_vit cublasMMWrapper memory_utils -lcudnn -lm)