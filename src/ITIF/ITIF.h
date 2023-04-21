#pragma once

#include <vector>
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
class ITIF {
    private:
        int min_batch_size_;
        int min_seq_len_;
        int max_batch_size_;
        int max_seq_len_;
        int m_head_num_;
        int m_size_per_head_;
        int m_inter_size_;
        int hidden_units_;
        float*** batch_result;
        float** gemm_result;
        int gemm_num = 4;
        cudaStream_t stream_;
        cublasMMWrapper* cublas_wrapper_;
        IAllocator allocator_;
        float** weights_ptr = nullptr;
        float** post_weights_ptr = nullptr;
        float** hA = nullptr;
        float** hB = nullptr;
        float** batch_qkv_kernel_ptr_ = nullptr;
        float** batch_qkv_input_ptr_ = nullptr;
        float** batch_qkv_buf_ptr_ = nullptr;
        float* q_buf_ = nullptr;
        float* k_buf_ = nullptr;
        float* v_buf_ = nullptr;
        float* input_addr  = nullptr;
        void buildtensor();
        void freetensor();
    public: 
        ITIF(int min_batch_size,
        int min_seq_len,
        int max_batch_size,
        int max_seq_len,
        int m_head_num,
        int m_size_per_head,
        int m_inter_size,
        int m_hidden_units,
        cudaStream_t stream_,
        cublasMMWrapper* cublas_wrapper_,
        IAllocator allocator_);
        ~ITIF();

        float performGemmTest(int batch_size, int seq_len);
        float performBatchedGemmTest(int gemm_num,int batch_size, int seq_len);

};

struct Node {
    int tenant_id;
    int length;
    int size;
    int batch_id;
    int batched_gemm_id;
    float* data;
};
std::list<Node> request_batching(const std::vector<Node>& request_list, const float** gemm_profiling_results, const float** batched_gemm_profiling_results, int num_tenants, std::list<Node>& linked_list);