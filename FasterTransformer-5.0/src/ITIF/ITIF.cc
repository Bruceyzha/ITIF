#include "src/ITIF/ITIF.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <list>
#include <algorithm>
#include <iterator>

namespace fastertransformer {
// namespace fastertransformer
ITIF::ITIF(int min_batch_size,
           int min_seq_len,
           int max_batch_size,
           int max_seq_len,
           int m_head_num,
           int m_size_per_head,
           int m_inter_size,
           int m_hidden_units,
           cudaStream_t stream,
           cublasMMWrapper* cublas_wrapper,
           IAllocator allocator):
    min_batch_size_(min_batch_size),
    min_seq_len_(min_seq_len),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    m_head_num_(m_head_num),
    m_size_per_head_(m_size_per_head),
    m_inter_size_(m_inter_size),
    hidden_units_(m_hidden_units),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator)
{
    buildtensor();
    int h_token_num = max_batch_size * max_seq_len;
    size_t size = max_batch_size_ * max_seq_len_ * m_head_num_ * m_size_per_head_;
    device_malloc((void**)&input_addr, size * sizeof(float) * gemm_num);
    cudaMallocHost(&hA, sizeof(float*) * 3 * (3 * gemm_num + 1));
    cudaMallocHost(&hB, sizeof(float*) * 3 * (gemm_num + 1));
    for (int i = 0; i < 3 * (3 * gemm_num + 1); i++) {
        if (i < gemm_num) {
            hA[i] = weights_ptr[0] + i * hidden_units_ * hidden_units_;
        }
        else if (i >= gemm_num && i < 2 * gemm_num) {
            hA[i] = weights_ptr[2] + (i - gemm_num) * hidden_units_ * hidden_units_;
        }
        else if (i >= 2 * gemm_num && i < 3 * gemm_num) {
            hA[i] = weights_ptr[4] + (i - 2 * gemm_num) * hidden_units_ * hidden_units_;
        }
        else if (i == 3 * gemm_num) {
            hA[i] = nullptr;
        }
        else if (i >= 3 * gemm_num + 1 && i < 6 * gemm_num + 1) {
            hA[i] = input_addr + ((i - (3 * gemm_num + 1)) % gemm_num) * size;
        }
        else if (i == 6 * gemm_num + 1) {
            hA[i] = nullptr;
        }
        else if (i >= 6 * gemm_num + 2 && i < 7 * gemm_num + 2) {
            hA[i] = q_buf_ + (i - (6 * gemm_num + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else if (i >= 7 * gemm_num + 2 && i < 8 * gemm_num + 2) {
            hA[i] = k_buf_ + (i - (7 * gemm_num + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else if (i >= 8 * gemm_num + 2 && i < 9 * gemm_num + 2) {
            hA[i] = v_buf_ + (i - (8 * gemm_num + 2)) * (max_batch_size_ * max_seq_len_ * hidden_units_);
        }
        else {
            hA[i] = nullptr;
        }
    }
    batch_qkv_input_ptr_ = batch_qkv_kernel_ptr_ + 3 * gemm_num + 1;
    batch_qkv_buf_ptr_ = batch_qkv_kernel_ptr_ + 2 * (3 * gemm_num + 1);
    cudaError_t err1 = cudaMemcpyAsync(
        batch_qkv_kernel_ptr_, hA, sizeof(float*) * 3 * (3 * gemm_num + 1), cudaMemcpyHostToDevice, stream);

    int seq_step = 32;
    batch_result = new float**[3 * gemm_num];
    for (size_t i = 0; i < 3 * gemm_num; i++) {
        batch_result[i] = new float*[max_batch_size_];
        for (size_t j = 0; j < max_batch_size_; i++) {
            batch_result[i][j] = new float[(max_seq_len_ - min_seq_len_) / seq_step];
            for (size_t k = 0; k < (max_seq_len_ - min_seq_len_) / seq_step; k++) {
                batch_result[i][j][k] = BatchedGemmTest(i, j, k);
            }
        }
    }
    gemm_result = new float*[max_batch_size_];
    for (size_t i = 0; i < max_batch_size_; i++) {
        gemm_result[i] = new float[(max_seq_len_ - min_seq_len_) / seq_step];
        for (size_t j = 0; j < (max_seq_len_ - min_seq_len_) / seq_step; j++) {
            gemm_result[i][j] = GemmTest(i, j);
        }
    }
}
ITIF::~ITIF()
{
    freetensor();
    for (size_t i = 0; i < gemm_num; i++) {
        for (size_t j = 0; j < max_batch_size_; i++) {
            delete[] batch_result[i][j];
        }
        delete[] batch_result[i];
    }
    delete[] batch_result;
    for (size_t i = 0; i < max_batch_size_; i++) {
        delete[] gemm_result[i];
    }
    delete[] gemm_result;
}
ITIF::buildtensor()
{
    weights_ptr = new float*[7];
    deviceMalloc(&weights_ptr[0], gemm_num * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[1], gemm_num * hidden_units_);
    deviceMalloc(&weights_ptr[2], gemm_num * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[3], gemm_num * hidden_units_);
    deviceMalloc(&weights_ptr[4], gemm_num * hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], gemm_num * hidden_units_);
    deviceMalloc(&weights_ptr[6], gemm_num * hidden_units_ * hidden_units_);
    q_buf_ =
        (float*)allocator->malloc(sizeof(float) * 3 * max_batch_size_ * max_seq_len_ * hidden_units_ * gemm_num, false);
    k_buf_ = q_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_ * gemm_num;
    v_buf_ = k_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_ * gemm_num;
    qkv_buf_2_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * gemm_num, false);
    attn_out_buf_ =
        (float*)allocator->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * hidden_units_ * gemm_num, false);
}
ITIF::freetensor()
{
    allocator->free(q_buf_);
    for (size_t i = 0; i < 7; i++) {
        deviceFree(weights_ptr[i]);
    }
    delete[] weights_ptr;
}
float ITIF::performGemmTest(int batch_size, int seq_len)
{
    int m = batch_size * seq_len;
    int k = m_head_num_ * m_size_per_head_;
    int n = hidden_units_;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights_ptr[0], n, input_addr, k, q_buf_, n);
    gettimeofday(&end_time, 0);
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    return seconds + 1e-6 * microseconds;
}

float ITIF::BatchedGemmTest(int gemm_num, int batch_size, int seq_len)
{
    int m = batch_size * seq_len;
    int k = m_head_num_ * m_size_per_head_;
    int n = hidden_units_;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
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
                                   gemm_num);
}
gettimeofday(&end_time, 0);
seconds = end_time.tv_sec - start_time.tv_sec;
microseconds = end_time.tv_usec - start_time.tv_usec;
return seconds + 1e-6 * microseconds;
}  // namespace fastertransformer

struct Request {
    int tenant_id;
    int length;
};

struct LinkedNode {
    int tenant;
    int index;
    int length;
};

void update_linked_list(LinkedNode*& linked_list, const std::vector<std::vector<int>>& mini_batch_idx_lists) {
    LinkedNode* new_head = nullptr;
    LinkedNode* new_tail = nullptr;
    
    for (const auto& tenant_mini_batch_idx_list : mini_batch_idx_lists) {
        for (int idx : tenant_mini_batch_idx_list) {
            LinkedNode* current = linked_list;
            LinkedNode* prev = nullptr;
            
            for (int i = 0; i < idx; ++i) {
                prev = current;
                current = current->next;
            }
            
            if (prev) {
                prev->next = current->next;
            } else {
                linked_list = linked_list->next;
            }
            
            current->next = nullptr;
            
            if (new_tail) {
                new_tail->next = current;
                new_tail = new_tail->next;
            } else {
                new_head = current;
                new_tail = new_head;
            }
        }
    }
    
    linked_list = new_head;
}

std::list<Node> request_batching(const std::vector<Node>& request_list, const float** gemm_profiling_results, const float** batched_gemm_profiling_results, int num_tenants, std::list<Node>& linked_list) {
    int N = request_list.size();

    // Tenant Request Batching with DP
    std::list<Node> linked_list;
    for (const Node& node : request_list) {
        linked_list.push_back(node);
    }
    std::vector<std::vector<float>> states(num_tenants, std::vector<float>(N + 1, 0));
    std::vector<std::list<int>> mini_batch(num_tenants);

    for (int t = 0; t < num_tenants; t++) {
        int batch_id = 0;
        for (int idx = 0; idx < N; idx++) {
            int j = idx - 1;
            int start_id = idx - 1;
            int seq_len = request_list[t][idx - 1].length;

            float optimal = gemm_profiling_results[seq_len][1] + states[t][j];
            while (j > 0) {
                float tmp = states[t][j - 1] + gemm_profiling_results[seq_len][idx - j + 1];
                if (tmp < optimal) {
                    optimal = tmp;
                    start_id = j - 1;
                }
                j--;
            }

            states[t][idx] = optimal;
            mini_batch[t].push_back(start_id);
            Node new_node = request_list[start_id];
            new_node.batch_id = batch_id;
            mini_batch[t].push_back(new_node);
            batch_id++;
        }
    }

    // Update the linked_list with the mini_batch_idx_lists
    for (int t = 0; t < num_tenants; t++) {
        auto it = linked_list.begin();
        for (int idx : mini_batch[t]) {
            auto target_iter = linked_list.begin();
            std::advance(target_iter, idx);

            Node temp_node = *it;
            linked_list.erase(it);
            linked_list.insert(target_iter, temp_node);

            ++it;
        }
    }

    // Compute-bound Operator Batching with DP
    std::vector<float> state(M, 0);
    std::vector<int> start_id_list(M, 0);
    int batched_gemm_id = 0;
    for (int idx = 0; idx < M; idx++) {
        int batched_idx = idx;
        int j = idx - 1;
        int start_id = idx - 1;
        int seq_len = mini_batch[idx].length;

        float optimal = gemm_profiling_results[seq_len][mini_batch[idx].size] + state[j];
        while (mini_batch[batched_idx].size / mini_batch[idx].size == 0) {
            batched_idx++;
        }

        if (batched_idx - idx == 1) {
            state[idx] = optimal;
            start_id_list[idx] = start_id;
        } else {
            for (int i = 0; i < batched_idx; i++) {
                j = i - 1;
                start_id = i - 1;
                seq_len = mini_batch[i].length;
                optimal = gemm_profiling_results[seq_len][mini_batch[i].size] + state[j];

                while (j > 0) {
                    float tmp = state[j - 1] + batched_gemm_profiling_results[seq_len][i - j + 1][mini_batch[i].size];
                    if (tmp < optimal) {
                        optimal = tmp;
                        start_id = j - 1;
                    }
                    j--;
                }
                state[idx] = optimal;
                start_id_list[idx] = start_id;
            }
        }
        linked_list_it->batched_gemm_id = batched_gemm_id;
        batched_gemm_id++;
    }

    // Update the linked list based on the start_id_list
    auto it = linked_list.begin();
    for (int i = 0; i < M; ++i) {
        int start_id = start_id_list[i];
        if (start_id >= 0) {
            auto start_iter = linked_list.begin();
            std::advance(start_iter, start_id);

            Node temp_node = *it;
            linked_list.erase(it);
            linked_list.insert(start_iter, temp_node);
        }
        ++it;
    }
    return linked_list;
}