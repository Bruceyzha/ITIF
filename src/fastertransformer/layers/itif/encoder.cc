#include "src/fastertransformer/layers/itif/encoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/changed_layer_norm.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
namespace fastertransformer {

encoder::encoder(size_t max_batch_size,
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
                 float* output_addr):
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    num_model_(num_model)
{
    printf("build encoder\n");
    buildWeights();
    allocateBuffer();
    int h_token_num = max_batch_size * max_seq_len;
    size_t size = max_batch_size_ * max_seq_len_ * head_num_ * size_per_head_;
    cudaMallocHost(&hA, sizeof(float*) * 3 * (3 * num_model_ + 1));
    cudaMallocHost(&hB, sizeof(float*) * 3 * (num_model_ + 1));
    cudaMallocHost(&hC, sizeof(float*) * 3 * (num_model_ + 1));
    cudaMallocHost(&hD, sizeof(float*) * 3 * (num_model_ + 1));

    for (int i = 0; i < 3 * (3 * num_model_ + 1); i++) {
        if (i < num_model_) {
            hA[i] = weights_ptr[0] + i * hidden_units_ * hidden_units_;
        }
        else if (i >= num_model_ && i < 2 * num_model_) {
            hA[i] = weights_ptr[2] + (i - num_model_) * hidden_units_ * hidden_units_;
        }
        else if (i >= 2 * num_model_ && i < 3 * num_model_) {
            hA[i] = weights_ptr[4] + (i - 2 * num_model_) * hidden_units_ * hidden_units_;
        }
        else if (i == 3 * num_model_) {
            hA[i] = nullptr;
        }
        else if (i >= 3 * num_model_ + 1 && i < 6 * num_model_ + 1) {
            hA[i] = input_addr + ((i - (3 * num_model_ + 1)) % num_model_) * size;
        }
        else if (i == 6 * num_model_ + 1) {
            hA[i] = nullptr;
        }
        else if (i >= 6 * num_model_ + 2 && i < 7 * num_model_ + 2) {
            hA[i] = q_buf_ + (i - (6 * num_model_ + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else if (i >= 7 * num_model_ + 2 && i < 8 * num_model_ + 2) {
            hA[i] = k_buf_ + (i - (7 * num_model_ + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else if (i >= 8 * num_model_ + 2 && i < 9 * num_model_ + 2) {
            hA[i] = v_buf_ + (i - (8 * num_model_ + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else {
            hA[i] = nullptr;
        }
        // printf("%p\n",hA[i]);
    }
    batch_qkv_input_ptr_ = batch_qkv_kernel_ptr_ + 3 * num_model_ + 1;
    batch_qkv_buf_ptr_ = batch_qkv_kernel_ptr_ + 2 * (3 * num_model_ + 1);
    cudaError_t err1 = cudaMemcpyAsync(
        batch_qkv_kernel_ptr_, hA, sizeof(float*) * 3 * (3 * num_model_ + 1), cudaMemcpyHostToDevice, stream[0]);

    for (int i = 0; i < 3 * (num_model_ + 1); i++) {
        if (i < num_model_) {
            hB[i] = weights_ptr[6] + i * hidden_units_ * hidden_units_;
        }
        else if (i == num_model_) {
            hB[i] = nullptr;
        }
        else if (i >= num_model_ + 1 && i < 2 * num_model_ + 1) {
            hB[i] = qkv_buf_2_ + (i - (num_model_ + 1)) * size;
        }
        else if (i == 2 * num_model_ + 1) {
            hB[i] = nullptr;
        }
        else if (i >= 2 * num_model_ + 2 && i < 3 * num_model_ + 2) {
            hB[i] = attn_out_buf_ + (i - (2 * num_model_ + 2)) * size;
        }
        else {
            hB[i] = nullptr;
        }
        // printf("%p %p\n",hB[i],hB);
    }
    batch_atten_out_input_ptr_ = batch_atten_out_kernel_ptr_ + num_model_ + 1;
    batch_atten_out_buf_ptr_ = batch_atten_out_kernel_ptr_ + 2 * (num_model_ + 1);
    cudaError_t err = cudaMemcpyAsync(
        batch_atten_out_kernel_ptr_, hB, sizeof(float*) * 3 * (num_model_ + 1), cudaMemcpyHostToDevice, stream[0]);

    for (int i = 0; i < 3 * (num_model_ + 1); i++) {
        if (i < num_model_) {
            hC[i] = weights_ptr[10] + i * hidden_units_ * inter_size_;
        }
        else if (i == num_model_) {
            hC[i] = nullptr;
        }
        else if (i >= num_model_ + 1 && i < 2 * num_model_ + 1) {
            hC[i] = attn_out_buf_ + (i - (num_model_ + 1)) * size;
        }
        else if (i == 2 * num_model_ + 1) {
            hC[i] = nullptr;
        }
        else if (i >= 2 * num_model_ + 2 && i < 3 * num_model_ + 2) {
            hC[i] = inter_buf_ + (i - (2 * num_model_ + 2)) * size;
        }
        else {
            hC[i] = nullptr;
        }
    }
    batch_ffn_input_ptr_ = batch_ffn_kernel_ptr_ + num_model_ + 1;
    batch_ffn_buf_ptr_ = batch_ffn_kernel_ptr_ + 2 * (num_model_ + 1);
    cudaError_t err3 = cudaMemcpyAsync(
        (void*)batch_ffn_kernel_ptr_, hC, sizeof(float*) * 3 * (num_model_ + 1), cudaMemcpyHostToDevice, stream[0]);

    for (int i = 0; i < 3 * (num_model_ + 1); i++) {
        if (i < num_model_) {
            hD[i] = weights_ptr[12] + i * hidden_units_ * inter_size_;
        }
        else if (i == num_model_) {
            hD[i] = nullptr;
        }
        else if (i >= num_model_ + 1 && i < 2 * num_model_ + 1) {
            hD[i] = inter_buf_ + (i - (num_model_ + 1)) * h_token_num * inter_size_;
        }
        else if (i == 2 * num_model_ + 1) {
            hD[i] = nullptr;
        }
        else if (i >= 2 * num_model_ + 2 && i < 3 * num_model_ + 2) {
            hD[i] = output_addr + (i - (2 * num_model_ + 2)) * size;
        }
        else {
            hD[i] = nullptr;
        }
    }
    batch_ffn_out_input_ptr_ = batch_ffn_out_kernel_ptr_ + num_model_ + 1;
    batch_ffn_out_buf_ptr_ = batch_ffn_out_kernel_ptr_ + 2 * (num_model_ + 1);
    cudaMemcpyAsync(
        (void*)batch_ffn_out_kernel_ptr_, hD, sizeof(float*) * 3 * (num_model_ + 1), cudaMemcpyHostToDevice, stream[0]);
}
encoder::~encoder()
{
    freeBuffer();
    freeWeights();
}

void encoder::buildWeights()
{
    weights_ptr = new float*[16];
    deviceMalloc(&weights_ptr[0], num_model_ * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[1], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[2], num_model_ * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[3], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[4], num_model_ * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[6], num_model_ * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[7], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[8], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[9], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[10], num_model_ * hidden_units_ * inter_size_);
    deviceMalloc(&weights_ptr[11], num_model_ * inter_size_);
    deviceMalloc(&weights_ptr[12], num_model_ * inter_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[13], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[14], num_model_ * hidden_units_);
    deviceMalloc(&weights_ptr[15], num_model_ * hidden_units_);
    post_weights_ptr = new float*[2];
    deviceMalloc(&post_weights_ptr[0], num_model_ * hidden_units_);
    deviceMalloc(&post_weights_ptr[1], num_model_ * hidden_units_);
}
void encoder::freeWeights()
{
    // for (int i = 0; i < 16; i++) {
    //     deviceFree(weights_ptr[i]);
    // }
    // free(weights_ptr);
}
void encoder::allocateBuffer() 
{
    q_buf_ = (float*)allocator->malloc(sizeof(float) * 3 * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_,
                                       false);
    k_buf_ = q_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_;
    v_buf_ = k_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_;
    q_buf_2_ = (float*)allocator->malloc(
        sizeof(float) * 3 * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_, false);
    k_buf_2_ = q_buf_2_ + max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_;
    v_buf_2_ = k_buf_2_ + max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_;
    qk_buf_ = (float*)allocator->malloc(
        sizeof(float) * max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ * num_model_, false);
    qkv_buf_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_, false);
    batch_qkv_kernel_ptr_ = (float**)allocator->malloc(sizeof(float*) * 3 * (3 * num_model_ + 1), false);
    qkv_buf_2_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_, false);
    batch_atten_out_kernel_ptr_ = (float**)allocator->malloc(sizeof(float*) * 3 * (num_model_ + 1), false);
    inter_buf_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * num_model_ * max_seq_len_ * inter_size_, false);
    batch_ffn_kernel_ptr_ = (float**)allocator->malloc(sizeof(float*) * 3 * (num_model_ + 1), false);
    batch_ffn_out_kernel_ptr_ = (float**)allocator->malloc(sizeof(float*) * 3 * (num_model_ + 1), false);
    attn_out_buf_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_, false);
    normed_attn_out_buf_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * num_model_, false);
}

void encoder::freeBuffer()
{
    allocator->free(q_buf_);
    allocator->free(k_buf_);
    allocator->free(v_buf_);

    allocator->free(q_buf_2_);
    allocator->free(qk_buf_);
    allocator->free(qkv_buf_);
    allocator->free(batch_qkv_kernel_ptr_);
    allocator->free(qkv_buf_2_);
    allocator->free(batch_atten_out_kernel_ptr_);
    allocator->free(inter_buf_);
    allocator->free(batch_ffn_kernel_ptr_);
    allocator->free(batch_ffn_out_kernel_ptr_);
    allocator->free(attn_out_buf_);
    allocator->free(normed_attn_out_buf_);
}

void encoder::forward(std::vector<Tensor>* input_tensors,
                      const std::vector<Tensor>* output_tensors,
                      const std::vector<Tensor>* model_offsets,
                      float* attention_mask,
                      int* check_array)
{
    // input_tensors:
    //      input_query [batch, seqlen, hidden]
    //      sequence_length [batch]
    // output tensors:
    //      output hidden state [batch, seqlen, hidden]

    const size_t batch_size = max_batch_size_;
    const size_t seq_len = input_tensors->at(0).shape[1];
    int* model_offset = (int*)model_offsets->at(0).data;
    size_t size = batch_size * seq_len * head_num_ * size_per_head_;
    int d_model = head_num_ * size_per_head_;
    int h_token_num = batch_size * seq_len;
    float* input_tensor = (float*)input_tensors->at(0).data;
    float* output_tensor = (float*)output_tensors->at(0).data;
    //      // attetion
    int m = batch_size * seq_len;
    int k = d_model;
    int n = hidden_units_;
    if (check_array[0] == 1) {
        cublas_wrapper[0]->batchedGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    (const void* const*)batch_qkv_kernel_ptr_,
                                    n,
                                    (const void* const*)batch_qkv_input_ptr_,
                                    k,
                                    (void* const*)batch_qkv_buf_ptr_,
                                    n,
                                    3 * num_model_);
    }
    else {
        for (int j = 0; j < num_model_; j++) {
            cublas_wrapper[0]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    weights_ptr[0] + j * hidden_units_ * hidden_units_,
                                    n,
                                    input_tensor + j * size,
                                    k,
                                    q_buf_ + j * max_batch_size_ * max_seq_len_ * hidden_units_,
                                    n);

            cublas_wrapper[1]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    weights_ptr[2] + j * hidden_units_ * hidden_units_,
                                    n,
                                    input_tensor + j * size,
                                    k,
                                    k_buf_ + j * max_batch_size_ * max_seq_len_ * hidden_units_,
                                    n);

            cublas_wrapper[2]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    weights_ptr[4] + j * hidden_units_ * hidden_units_,
                                    n,
                                    input_tensor + j * size,
                                    k,
                                    v_buf_ + j * max_batch_size_ * max_seq_len_ * hidden_units_,
                                    n);
        }
    }
    invokeAddQKVBiasTranspose(q_buf_2_,
                                  k_buf_2_,
                                  v_buf_2_,
                                  q_buf_,
                                  weights_ptr[1],
                                  k_buf_,
                                  weights_ptr[3],
                                  v_buf_,
                                  weights_ptr[5],
                                  batch_size * num_model_,
                                  seq_len,
                                  head_num_,
                                  size_per_head_,
                                  num_model_,
                                  model_offset,
                                  stream[0]);
    float scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * 1.0f);
    for (int i = 0; i < num_model_; i++) {
        cublas_wrapper[i]->stridedBatchedGemm(CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              seq_len,
                                              seq_len,
                                              size_per_head_,
                                              k_buf_2_ + i * max_batch_size_ * max_seq_len_ * hidden_units_,
                                              size_per_head_,
                                              seq_len * size_per_head_,
                                              q_buf_2_ + i * max_batch_size_ * max_seq_len_ * hidden_units_,
                                              size_per_head_,
                                              seq_len * size_per_head_,
                                              qk_buf_ + i * max_batch_size_ * max_seq_len_ * hidden_units_,
                                              seq_len,
                                              seq_len * seq_len,
                                              batch_size * head_num_,
                                              scalar);
    }
    invokeMaskedSoftMax(qk_buf_, qk_buf_, attention_mask, batch_size * num_model_, seq_len, head_num_, 1.0f, stream[0]);
    for (int i = 0; i < num_model_; i++) {
        cublas_wrapper[i]->stridedBatchedGemm(CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              size_per_head_,
                                              seq_len,
                                              seq_len,
                                              v_buf_2_ + i * max_batch_size_ * max_seq_len_ * hidden_units_,
                                              size_per_head_,
                                              seq_len * size_per_head_,
                                              qk_buf_ + i * max_batch_size_ * max_seq_len_ * hidden_units_,
                                              seq_len,
                                              seq_len * seq_len,
                                              qkv_buf_,
                                              size_per_head_,
                                              seq_len * size_per_head_,
                                              batch_size * head_num_);
    }
    invokeTransposeQKV(qkv_buf_2_, qkv_buf_, batch_size * num_model_, seq_len, head_num_, size_per_head_, stream[0]);
    k = hidden_units_;
    n = d_model;
    if (check_array[1] == 1) {
        cublas_wrapper[0]->batchedGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    (const void* const*)batch_atten_out_kernel_ptr_,
                                    n,
                                    (const void* const*)batch_atten_out_input_ptr_,
                                    k,
                                    (void* const*)batch_atten_out_buf_ptr_,
                                    n,
                                    num_model_);
    }
    else {
        for (int i = 0; i < num_model_; i++) {
            cublas_wrapper[i]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    weights_ptr[6] + i * hidden_units_ * hidden_units_,
                                    n,
                                    qkv_buf_2_ + i * size,
                                    k,
                                    attn_out_buf_ + i * size,
                                    n);
        }
    }
    invokeAddBiasResidualLayerNorm_modified<float>(attn_out_buf_,
                                                       input_tensor,
                                                       weights_ptr[14],
                                                       weights_ptr[15],
                                                       weights_ptr[7],
                                                       h_token_num*num_model_,
                                                       hidden_units_,
                                                       num_model_,
                                                       model_offset,
                                                       stream[0]);
    m = h_token_num;
    if (check_array[1] == 1) {
        cublas_wrapper[0]->batchedGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    inter_size_,
                                    m,
                                    hidden_units_,
                                    (const void* const*)batch_ffn_kernel_ptr_,
                                    inter_size_,
                                    (const void* const*)batch_ffn_input_ptr_,
                                    hidden_units_,
                                    (void* const*)batch_ffn_buf_ptr_,
                                    inter_size_,
                                    num_model_);
    }
    else {
        for (int i = 0; i < num_model_; i++) {
            cublas_wrapper[i]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    inter_size_,
                                    m,
                                    hidden_units_,
                                    weights_ptr[10] + i * hidden_units_ * inter_size_,
                                    inter_size_,
                                    normed_attn_out_buf_ + i * size,
                                    hidden_units_,
                                    inter_buf_ + i * inter_size_ * h_token_num,
                                    inter_size_);
        }
    }
    invokeAddBiasRelu(inter_buf_, weights_ptr[11], m, inter_size_*num_model_, num_model_, model_offset, stream[0]);
    if (check_array[1] == 1) {
        cublas_wrapper[0]->batchedGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m,
                                    inter_size_,
                                    (const void* const*)batch_ffn_out_kernel_ptr_,
                                    hidden_units_,
                                    (const void* const*)batch_ffn_out_input_ptr_,
                                    inter_size_,
                                    (void* const*)batch_ffn_out_buf_ptr_,
                                    hidden_units_,
                                    num_model_);
    }
    else {
        for (int i = 0; i < num_model_; i++) {
            cublas_wrapper[i]->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m,
                                    inter_size_,
                                    weights_ptr[12] + i * hidden_units_ * inter_size_,
                                    hidden_units_,
                                    inter_buf_ + i * h_token_num * inter_size_,
                                    inter_size_,
                                    output_tensor + i * size,
                                    hidden_units_);
        }
    }
    invokeAddBiasResidual_modified(
            output_tensor, attn_out_buf_, weights_ptr[13], h_token_num*num_model_, hidden_units_, num_model_, model_offset, stream[0]);
   
}

bool encoder::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }
    return true;
}

bool encoder::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ < seq_len) {
        max_seq_len_ = seq_len;
    }
    return true;
}

}  // namespace fastertransformer