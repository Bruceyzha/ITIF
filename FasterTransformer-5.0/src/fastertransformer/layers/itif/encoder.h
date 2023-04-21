#pragma once

#include <vector>
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/changed_layer_norm.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"

namespace fastertransformer {

class encoder{
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t num_model_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t hidden_units_;
    size_t num_layer_;
    int* model_offset_;
    cudaStream_t* stream;
    cublasMMWrapper** cublas_wrapper;
    IAllocator* allocator;
    void allocateBuffer();
    void freeBuffer();
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    void buildWeights();
    void freeWeights();

protected:
    // model params
    float** weights_ptr = nullptr;
    float** post_weights_ptr = nullptr;
    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
    float* q_buf_2_ = nullptr;
    float* k_buf_2_ = nullptr;
    float* v_buf_2_ = nullptr;
    float* qk_buf_ = nullptr;
    float* qkv_buf_ = nullptr;
    float** batch_qkv_kernel_ptr_ = nullptr;
    float** batch_qkv_input_ptr_ = nullptr;
    float** batch_qkv_buf_ptr_ = nullptr;
    float* qkv_buf_2_ = nullptr;
    float** batch_atten_out_kernel_ptr_ = nullptr;
    float** batch_atten_out_input_ptr_  = nullptr;
    float** batch_atten_out_buf_ptr_ = nullptr;
    float* inter_buf_ = nullptr;
    float** batch_ffn_kernel_ptr_ = nullptr;
    float** batch_ffn_input_ptr_ = nullptr;
    float** batch_ffn_buf_ptr_ = nullptr;
    float** batch_ffn_out_kernel_ptr_ = nullptr;
    float** batch_ffn_out_input_ptr_ = nullptr;
    float** batch_ffn_out_buf_ptr_ = nullptr;
    //float* attention_mask = nullptr;
    float* attn_out_buf_ = nullptr;
    float* normed_attn_out_buf_ = nullptr;
    float** hA = nullptr;
    float** hB = nullptr;
    float** hC = nullptr;
    float** hD = nullptr;

public:
    encoder(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 size_t num_layer,
                 size_t num_model,
                 cudaStream_t* stream,
                 cublasMMWrapper** cublas_wrapper,
                 IAllocator* allocator,
                 bool gemm_test,
                 int* check_array,
                 float* input_addr,
                 float* output_addr);


    ~encoder();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<Tensor>* model_offsets,
                 float* attention_mask,
                 int* check_array);
};

}  // namespace fastertransformer