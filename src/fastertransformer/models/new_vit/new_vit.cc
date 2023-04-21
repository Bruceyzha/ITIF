#include "src/fastertransformer/models/new_vit/new_vit.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/changed_layer_norm.h"
#include "src/fastertransformer/kernels/vit_kernels.h"

namespace fastertransformer {

void ViTTransformer::initialize()
{
    FT_LOG_DEBUG("img_size: %lu, patch_size:%lu\n"
                 "batch_size:%lu, chn_num  : %lu\n"
                 "seq_len   :%lu, embed_dim: %lu\n"
                 "head_num  :%lu, head_dim : %lu\n"
                 "inter_size:%lu, num_layer: %lu\n"
                 "att_type  : %d, \n",
                 img_size_,
                 patch_size_,
                 max_batch_size_,
                 chn_num_,
                 max_seq_len_,
                 embed_dim_,
                 head_num_,
                 head_dim_,
                 inter_size_,
                 num_layer_);
    if (img_size_ % patch_size_ != 0) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] IMG size & PITCH size missmatch. " << img_size_ << " % " << patch_size_ << " !=0 \n";
        throw std::runtime_error(buffer.str());
    }

    if (head_num_ * head_dim_ != embed_dim_) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] Embed size and head number mismatch. Embed_dim=" << embed_dim_
               << "; head_num*head_dim = "
               << "(" << head_num_ << "*" << head_dim_ << ")=" << head_num_ * head_dim_ << std::endl;
        throw std::runtime_error(buffer.str());
    }
}

ViTTransformer::ViTTransformer(size_t max_batch_size,
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
                               float* from_tensor):
    max_batch_size_(max_batch_size),
    img_size_(img_size),
    chn_num_(chn_num),
    patch_size_(patch_size),
    request_seq_len_(img_size * img_size / patch_size / patch_size + (with_cls_token ? 1 : 0)),
    max_seq_len_(0),
    embed_dim_(embed_dim),
    head_num_(head_num),
    head_dim_(embed_dim / head_num),
    inter_size_(inter_size),
    num_layer_(num_layer),
    with_cls_token_(with_cls_token),
    sm_(sm),
    q_scaling_(q_scaling),
    cudnn_handle_(cudnn_handle),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    num_models_(num_model)

{
    max_seq_len_ = request_seq_len_;
    initialize();
    buildWeights();
    allocateBuffer();
    Encoder_ = new encoder(max_batch_size_,
                           max_seq_len_,
                           head_num_,
                           head_dim_,
                           inter_size_,
                           num_layer_,
                           num_models_,
                           stream_,
                           cublas_wrapper_,
                           allocator_,
                           true,
                           check_array,
                           embed_buf_1_,
                           embed_buf_3_);
    patchEmbed(embed_buf_1_,
               from_tensor,
               weights_ptr[0],
               weights_ptr[1],
               weights_ptr[2],
               weights_ptr[3],
               max_batch_size_,
               224,
               patch_size_,
               max_seq_len_,
               3,
               embed_dim_);
}

ViTTransformer::~ViTTransformer()
{
    delete Encoder_;
    freeBuffer();
    freeWeights();
}

void ViTTransformer::buildWeights()
{
    weights_ptr = new float*[6];
    deviceMalloc(&weights_ptr[0], num_models_ * chn_num_ * patch_size_ * patch_size_ * embed_dim_);
    deviceMalloc(&weights_ptr[1], num_models_ * embed_dim_);
    deviceMalloc(&weights_ptr[2], num_models_ * embed_dim_);
    deviceMalloc(&weights_ptr[3], num_models_ * embed_dim_ * max_seq_len_);
    deviceMalloc(&weights_ptr[4], num_models_ * embed_dim_);
    deviceMalloc(&weights_ptr[5], num_models_ * embed_dim_);
}

void ViTTransformer::freeWeights()
{
    for (int i = 0; i < 6; i++) {
        deviceFree(weights_ptr[i]);
    }
    free(weights_ptr);
}

void ViTTransformer::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        embed_buf_1_ = (float*)allocator_->malloc(
            sizeof(float) * max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_*4, false);
        embed_buf_2_ =  embed_buf_1_ +  max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_;
        embed_buf_3_ =  embed_buf_2_ +  max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_;
        mask_buf_ =  embed_buf_3_ +  max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_;
        
        // embed_buf_2_ = (float*)allocator_->malloc(
        //     sizeof(float) * max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_, false);
        //     printf("addr %p\n",embed_buf_2_);
        // embed_buf_3_ = (float*)allocator_->malloc(
        //     sizeof(float) * max_batch_size_ * max_seq_len_ * embed_dim_ * num_models_, false);
        // mask_buf_ = (float*)allocator_->malloc(
        //     sizeof(float) * max_batch_size_ * max_seq_len_ * max_seq_len_ * num_models_, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_ * num_models_, false);
        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);

        trt_mha_padding_offset_ =
            (int*)allocator_->malloc(sizeof(int) * (2 * max_batch_size_ * num_models_ + 1), false);
        seq_len_vec_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * num_models_, false);

        // setSeqLenVec(max_batch_size_ * num_models_);
        // setDefaultMask(max_batch_size_ * num_models_);
        // setDefaultPaddingOffset(max_batch_size_ * num_models_);

        is_allocate_buffer_ = true;
    }
}

void ViTTransformer::freeBuffer()
{
    allocator_->free(embed_buf_1_);
    // allocator_->free(embed_buf_2_);
    // allocator_->free(embed_buf_3_);
    // allocator_->free(mask_buf_);
    allocator_->free(trt_mha_padding_offset_);
    allocator_->free(seq_len_vec_);
    allocator_->free(padding_offset_);
    allocator_->free(token_num_);

    is_allocate_buffer_ = false;
}

void ViTTransformer::forward(std::vector<Tensor>* output_tensors,
                             const std::vector<Tensor>* input_tensors,
                             const std::vector<Tensor>* model_offsets)
{
    // input_tensors:
    //      input_img, BCHW [batch, chn_num, img_size, img_size]
    // output tensors:
    //      output feature_map [batch, seq_len, embed_dim]

    const size_t input_batch_size = input_tensors->at(0).shape[0];
    const size_t input_chn_num = input_tensors->at(0).shape[1];
    const size_t input_img_size = input_tensors->at(0).shape[2];
    const size_t patch_resol = input_img_size / patch_size_;
    size_t seq_len = patch_resol * patch_resol + (with_cls_token_ ? 1 : 0);
    int* model_offset = (int*)model_offsets->at(0).data;

    const float* input = (const float*)input_tensors->at(0).data;
    float* output = (float*)output_tensors->at(0).data;
    float* encoder_input_ptr = embed_buf_1_;
    // preprocess (patches embedding, concat class embed and add pos embed)

    DataType data_type = getTensorType<float>();

    size_t h_token_num = max_batch_size_ * seq_len;
    // get offsets
    float* from_buf = encoder_input_ptr;
    float* norm_out_buf = embed_buf_2_;
    float* attn_out_buf = embed_buf_3_;
    float* encoder_out_buf = from_buf;

    for (uint i = 0; i < num_layer_; i++) {

        std::vector<Tensor> attn_input_tensors{Tensor{
            MEMORY_GPU, data_type, std::vector<size_t>{max_batch_size_ * num_models_, seq_len, embed_dim_}, from_buf}};
        std::vector<Tensor> attn_output_tensors{
            Tensor{MEMORY_GPU, data_type, std::vector<size_t>{max_batch_size_ * num_models_, seq_len}, attn_out_buf}};
        Encoder_->forward(&attn_input_tensors, &attn_output_tensors, model_offsets, mask_buf_, check_array);
    }

    invokeGeneralLayerNorm_modified(output,
                                    from_buf,
                                    weights_ptr[4],
                                    weights_ptr[5],
                                    h_token_num * num_models_,
                                    embed_dim_,
                                    num_models_,
                                    model_offset,
                                    stream_[0]);

    sync_check_cuda_error();
}

bool ViTTransformer::resetBatch(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }

    return true;
}

bool ViTTransformer::setSeqLenVec(size_t batch_size)
{
    int* seq_len_vec = new int[batch_size];
    for (int i = 0; i < batch_size; i++) {
        seq_len_vec[i] = request_seq_len_;
    }
    cudaMemcpy(seq_len_vec_, seq_len_vec, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    delete seq_len_vec;
    return true;
}

void ViTTransformer::setDefaultMask(size_t batch_size)
{
    invokeBuildEncoderAttentionMask(mask_buf_, seq_len_vec_, batch_size, max_seq_len_, stream_[0]);
}

void ViTTransformer::setDefaultPaddingOffset(size_t batch_size)
{
    invokeGetPaddingOffset(
        &nopad_token_num_, token_num_, padding_offset_, seq_len_vec_, batch_size, max_seq_len_, stream_[0]);
}

void ViTTransformer::patchEmbed(float* output,
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
                                const int embed_dim)
{
    float* tmp_buf = embed_buf_2_;
    
    int input_size = batch * img_size * img_size * in_chans;
    int output_size = batch * seq_len * embed_dim;
    int kernel_size = chn_num_ * patch_size * patch_size * embed_dim;
    for (int i = 0; i < num_models_; i++) {
        conv2d(tmp_buf + i * output_size,
               input + i * input_size,
               kernel + i * kernel_size,
               batch,
               img_size,
               img_size,
               in_chans,
               embed_dim,
               patch_size,
               patch_size,
               cudnn_handle_);
        int n = embed_dim;
        int s = seq_len;
        int m = batch * s;
        if (with_cls_token_) {
            FT_CHECK(cls_embed != nullptr);
            invokeAddBiasConcatClsTokenAddPosEmbed(tmp_buf + i * output_size,
                                                   output + i * output_size,
                                                   bias + i * embed_dim_,
                                                   cls_embed + i * embed_dim_,
                                                   pos_embed + i * embed_dim_ * max_seq_len_,
                                                   m,
                                                   n,
                                                   s,
                                                   stream_[i]);
        }
        else {
            invokeAddBiasAddPosEmbed(tmp_buf + i * output_size,
                                     bias + i * embed_dim_,
                                     pos_embed + i * embed_dim_ * max_seq_len_,
                                     m,
                                     n,
                                     s * n,
                                     stream_[i]);
        }
    }
}

}  // namespace fastertransformer