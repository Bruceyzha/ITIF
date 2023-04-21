#pragma once

#include <vector>

// #include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/FusedAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"
#include "src/fastertransformer/models/vit/ViTWeight.h"
#include "src/fastertransformer/utils/conv2d.h"
#include "src/fastertransformer/layers/kernel_slice/unfused_mha.h"
namespace fastertransformer {

class ViTTransformer {
private:
    size_t max_batch_size_ = 0;
    size_t img_size_ = 224;
    size_t chn_num_ = 3;
    size_t patch_size_ = 16;  // preproc patch size
    size_t max_seq_len_;
    size_t request_seq_len_;
    size_t embed_dim_;   // patch conv out units, size_per_head = embed_dim / head_num
    size_t head_num_;    // mha head num
    size_t head_dim_;    // mha head size
    size_t inter_size_;  // FF internal size
    size_t num_layer_;
    size_t nopad_token_num_;
    int num_models_;
    encoder* Encoder_;
    bool with_cls_token_;
    int sm_;
    int check_array[2] = {0,0};
    float q_scaling_;
    cudnnHandle_t cudnn_handle_;

    bool is_allocate_buffer_ = false;

    void buildWeights();
    void freeWeights();
    void allocateBuffer();
    void freeBuffer();
    bool resetBatch(size_t batch_size);
    bool setSeqLenVec(size_t batch_size);
    void setDefaultMask(size_t batch_size);
    void setDefaultPaddingOffset(size_t batch_size);
    void patchEmbed(float* output,
                    const float* input,
                    const float* kernel,
                    const float* bias,
                    const float* cls_embed,
                    const float* pos_embed,
                    const int batch,
                    const int img_size,
                    const int patch_size,
                    const int seq_len,
                    const int in_chans,
                    const int embed_dim);
    void initialize();


protected:
    // size_t* token_num_ = nullptr;
    float** weights_ptr = nullptr;
    float** post_weights_ptr = nullptr;
    float* embed_buf_1_ = nullptr;
    float* embed_buf_2_ = nullptr;
    float* embed_buf_3_ = nullptr;
    float* mask_buf_ = nullptr;
    int* trt_mha_padding_offset_ = nullptr;
    int* seq_len_vec_ = nullptr;
    int* padding_offset_ = nullptr;
    size_t* token_num_ = nullptr;
    cudaStream_t* stream_;
    cublasMMWrapper** cublas_wrapper_;
    IAllocator* allocator_;

public:
    ViTTransformer(size_t max_batch_size,
                   size_t img_size,
                   size_t chn_num,
                   size_t patch_size,
                   size_t embed_dim,
                   size_t head_num,
                   size_t inter_size,
                   size_t num_layer,
                   int num_model,
                   bool with_cls_token,
                   int sm,
                   float q_scaling,
                   cudaStream_t* stream,
                   cudnnHandle_t cudnn_handle,
                   cublasMMWrapper** cublas_wrapper,
                   IAllocator* allocator,
                   float* from_tensor);


    ~ViTTransformer();

    void
    forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors,const std::vector<Tensor>* model_offsets);
};

}  // namespace fastertransformer