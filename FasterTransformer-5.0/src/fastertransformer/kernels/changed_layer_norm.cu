#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/changed_layer_norm.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include <stdio.h>
namespace fastertransformer {
// // * Note that typename T is half2 or bfloat2 type
template<typename T>
__global__ void generalLayerNorm_modified(const T* __restrict input,
                                          const T* __restrict gamma,
                                          const T* __restrict beta,
                                          T* output,
                                          int m,
                                          int n,
                                          int num_models,
                                          int* model_offset)
{
    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }
    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + i >= model_offset[j]) {
                float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[j * n + i]);
                output[blockIdx.x * n + i] =
                    (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[j * n + i]))
                        + beta_val);
                break;
            }
        }
    }
}
template<typename T>
__global__ void generalLayerNorm(
    const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        output[blockIdx.x * n + i] =
            (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}
template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            int num_models,
                            int gamma_size,
                            int beta_size,
                            cudaStream_t* stream)
{
    dim3 grid(m);
    if (m / num_models > 1024) {
        dim3 grid(1024);
    }
    dim3 block(min(n, 1024));
    // printf("%d\n",block.x);
    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    /* should pay attention to the rsqrt precision*/
    // printf("invokeGeneralLayerNorm_modified thread require %d\n",gridDim*blockDim);
    int token_num = m / num_models;
    for (int i = 0; i < num_models; i++) {
        generalLayerNorm<T><<<grid, block, 0, stream[i]>>>(input + i * token_num * n,
                                                           gamma + i * gamma_size,
                                                           beta + i * beta_size,
                                                           out + i * token_num * n,
                                                           token_num,
                                                           n);
    }
}

template void invokeGeneralLayerNorm(float* out,
                                     const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     const int m,
                                     const int n,
                                     int num_models,
                                     int gamma_size,
                                     int beta_size,
                                     cudaStream_t* stream);
template<typename T>
void invokeGeneralLayerNorm_modified(T* out,
                                     const T* input,
                                     const T* gamma,
                                     const T* beta,
                                     const int m,
                                     const int n,
                                     int num_models,
                                     int* model_offset,
                                     cudaStream_t stream)
{
    dim3 grid(m);
    if (m > 1024) {
        dim3 grid(m);
    }
    dim3 block(min(n, 1024));
    // printf("%d\n",block.x);
    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    /* should pay attention to the rsqrt precision*/
    // printf("invokeGeneralLayerNorm_modified thread require %d\n",gridDim*blockDim);
    generalLayerNorm_modified<T>
        <<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n, num_models, model_offset);  // For gpt-3
    // printf("executed\n");
}
template void invokeGeneralLayerNorm_modified(float* out,
                                              const float* input,
                                              const float* gamma,
                                              const float* beta,
                                              const int m,
                                              const int n,
                                              int num_models,
                                              int* model_offset,
                                              cudaStream_t stream);

template<typename T>
__global__ void addBiasResidualPostLayerNormV2(T* out,
                                               const T* __restrict input,
                                               const T* __restrict bias,
                                               const T* __restrict gamma,
                                               const T* __restrict beta,
                                               int n,
                                               int num_models,
                                               int* model_offset)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n + col_id;
        for (int j = 0; j < num_models; j++) {
            if (id >= model_offset[j]) {
                local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id + j * n]));
                sum += local_out[i];
                break;
            }
        }
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n + col_id;
        for (int j = 0; j < num_models; j++) {
            if (id >= model_offset[j]) {
                out[id] = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id + j * n])
                              + (float)__ldg(&beta[col_id + j * n]));
                break;
            }
        }
    }
}
template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(T* out,
                                             const T* input,
                                             const T* bias,
                                             const T* gamma,
                                             const T* beta,
                                             int m,
                                             int n,
                                             int num_models,
                                             int* model_offset)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + idx >= model_offset[j]) {
                float local_out =
                    (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg(&bias[idx + j * n]));
                mean += local_out;
                // save local_out to local_out_cache to save some recompute
                local_out_cache[i] = local_out;
                idx += blockDim.x;
                break;
            }
        }
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + idx >= model_offset[j]) {
                out[blockIdx.x * n + idx] =
                    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx + j * n]))
                        + (float)(__ldg(&beta[idx + j * n])));
                idx += blockDim.x;
                break;
            }
        }
    }
}
template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(T* out,
                                                    const T* input,
                                                    const T* bias,
                                                    const T* gamma,
                                                    const T* beta,
                                                    int m,
                                                    int n,
                                                    int num_models,
                                                    int* model_offset)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + idx >= model_offset[j]) {
                float local_out =
                    (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg(&bias[idx + j * n]));
                mean += local_out;
                // save local_out to out to save some recompute
                out[blockIdx.x * n + idx] = local_out;
            }
        }
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + idx >= model_offset[j]) {
                float local_out = out[blockIdx.x * n + idx];
                out[blockIdx.x * n + idx] =
                    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx + j * n]))
                        + (float)(__ldg(&beta[idx + j * n])));
            }
        }
    }
}
template<typename T>
void invokeAddBiasResidualLayerNorm_modified(T* out,
                                             const T* input,
                                             const T* bias,
                                             const T* gamma,
                                             const T* beta,
                                             int m,
                                             int n,
                                             int num_models,
                                             int* model_offset,
                                             cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    // printf("invokeAddBiasResidualLayerNorm_modified thread require %d\n",gridDim*blockDim);
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2<T>
            <<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, n, num_models, model_offset);
    }
    else {
        block.x = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n, num_models, model_offset);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n, num_models, model_offset);
        }
        else {
            generalAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0>>>(out, input, bias, gamma, beta, m, n, num_models, model_offset);
        }
    }
}
template void invokeAddBiasResidualLayerNorm_modified(float* out,
                                                      const float* input,
                                                      const float* bias,
                                                      const float* gamma,
                                                      const float* beta,
                                                      int m,
                                                      int n,
                                                      int num_models,
                                                      int* model_offset,
                                                      cudaStream_t stream);

template<typename T>
__global__ void
addBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, int num_models, int* model_offset)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        for (int j = 0; j < num_models; j++) {
            if (blockIdx.x * n + col_index >= model_offset[j]) {
                T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index + j * n];
                output[blockIdx.x * n + col_index] =
                    output[blockIdx.x * n + col_index] + input[blockIdx.x * n + col_index] + bias_val;
                break;
            }
        }
    }
}
template<typename T>
void invokeAddBiasResidual_modified(T* output,
                                    const T* input,
                                    const T* bias,
                                    const int m,
                                    const int n,
                                    int num_models,
                                    int* model_offset,
                                    cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    // printf("invokeAddBiasResidual_modified thread require %d\n",gridDim*blockDim);
    addBiasResidual<<<grid, block, 0, stream>>>(output, input, bias, m, n, num_models, model_offset);
}
template<typename T>
__global__ void addBiasResidual(T* output, const T* input, const T* bias, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index];
        output[blockIdx.x * n + col_index] =
            output[blockIdx.x * n + col_index] + input[blockIdx.x * n + col_index] + bias_val;
    }
}
template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, int num_models, cudaStream_t* stream)
{
    int token_num = m /num_models;
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(token_num, blocks_per_row);
    dim3 block(min(n, 1024));
    int size = token_num * n;
    for(int i = 0; i < num_models; i++){
        addBiasResidual<<<grid, block, 0, stream[i]>>>(output + i * size, input + i * size, bias + i * n, token_num, n);
    }
}
template void invokeAddBiasResidual(float* output, const float* input, const float* bias, const int m, const int n, int num_models, cudaStream_t* stream);
template void invokeAddBiasResidual_modified(float* output,
                                             const float* input,
                                             const float* bias,
                                             const int m,
                                             const int n,
                                             int num_models,
                                             int* model_offset,
                                             cudaStream_t stream);

__global__ void addQKVBiasTranspose(float* q_out,
                                    float* k_out,
                                    float* v_out,
                                    float* __restrict q_in,
                                    const float* __restrict bias_q,
                                    float* __restrict k_in,
                                    const float* __restrict bias_k,
                                    float* __restrict v_in,
                                    const float* __restrict bias_v,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int num_models,
                                    const int* model_offset)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    // for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
    //     const int head_id = col_id / size_per_head;
    //     const int size_id = col_id % size_per_head;
    //     const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
    //                           + word_id * size_per_head + size_id;
    //     const int src_id = row_id * n + col_id;
    //     for(int j = 0; j < num_models; j++){
    //         if( src_id <= model_offset[j]){
    //             q_out[target_id] = __ldg(&q_in[src_id]);
    //             q_out[target_id] = q_out[target_id] + __ldg(&bias_q[col_id]);

    //             k_out[target_id] = __ldg(&k_in[src_id]);
    //             k_out[target_id] = k_out[target_id] + __ldg(&bias_k[col_id]);

    //             v_out[target_id] = __ldg(&v_in[src_id]);
    //             v_out[target_id] = v_out[target_id] + __ldg(&bias_v[col_id]);
    //             break;
    //         }
    //    }
    // }
    //  __syncthreads();
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;
        q_out[target_id] = __ldg(&q_in[src_id]);

        k_out[target_id] = __ldg(&k_in[src_id]);

        v_out[target_id] = __ldg(&v_in[src_id]);
    }
}
template<typename T>
__global__ void addQKVBiasTranspose(T* q_out,
                                    T* k_out,
                                    T* v_out,
                                    const T* __restrict q_in,
                                    const T* __restrict bias_q,
                                    const T* __restrict k_in,
                                    const T* __restrict bias_k,
                                    const T* __restrict v_in,
                                    const T* __restrict bias_v,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = __ldg(&q_in[src_id]);
        q_out[target_id] = q_out[target_id] + __ldg(&bias_q[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]);
        k_out[target_id] = k_out[target_id] + __ldg(&bias_k[col_id]);

        v_out[target_id] = __ldg(&v_in[src_id]);
        v_out[target_id] = v_out[target_id] + __ldg(&bias_v[col_id]);
    }
}
template<typename T>
__global__ void QKVTranspose(T* q_out,
                             T* k_out,
                             T* v_out,
                             const T* __restrict q_in,
                             const T* __restrict k_in,
                             const T* __restrict v_in,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = __ldg(&q_in[src_id]);
        k_out[target_id] = __ldg(&k_in[src_id]);
        v_out[target_id] = __ldg(&v_in[src_id]);
    }
}
void invokeAddQKVBiasTranspose(float* q_buf,
                               float* k_buf,
                               float* v_buf,
                               float* Q,
                               const float* bias_Q,
                               float* K,
                               const float* bias_K,
                               float* V,
                               const float* bias_V,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               const int num_models,
                               const int* model_offset,
                               cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    if (batch_size * seq_len > 1024) {
        dim3 grid(32, 32);
    }
    bool is_add_bias = bias_Q != nullptr;
    dim3 block(min(k, 512));
    addQKVBiasTranspose<<<grid, block, 0, stream>>>(q_buf,
                                                    k_buf,
                                                    v_buf,
                                                    Q,
                                                    bias_Q,
                                                    K,
                                                    bias_K,
                                                    V,
                                                    bias_V,
                                                    batch_size,
                                                    seq_len,
                                                    head_num,
                                                    size_per_head,
                                                    num_models,
                                                    model_offset);

    // sync_check_cuda_error();
}

void invokeAddQKVBiasTranspose(float* q_buf,
                               float* k_buf,
                               float* v_buf,
                               float* Q,
                               const float* bias_Q,
                               float* K,
                               const float* bias_K,
                               float* V,
                               const float* bias_V,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               const int num_models,
                               cudaStream_t* stream)
{
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    bool is_add_bias = bias_Q != nullptr;
    int hidden_units = head_num * size_per_head;
    int size = batch_size * seq_len * hidden_units;
    // if (sizeof(T) == 4 || k % 2 != 0) {
    dim3 block(min(k, 512));
    for (int i = 0; i < num_models; i++) {
        if (is_add_bias) {
            addQKVBiasTranspose<<<grid, block, 0, stream[i]>>>(q_buf + i * size,
                                                               k_buf + i * size,
                                                               v_buf + i * size,
                                                               Q + i * size,
                                                               bias_Q + i * hidden_units,
                                                               K + i * size,
                                                               bias_K + i * hidden_units,
                                                               V + i * size,
                                                               bias_V + i * hidden_units,
                                                               batch_size,
                                                               seq_len,
                                                               head_num,
                                                               size_per_head);
        }
        else {
            QKVTranspose<<<grid, block, 0, stream[i]>>>(q_buf + i * size,
                                                        k_buf + i * size,
                                                        v_buf + i * size,
                                                        Q + i * size,
                                                        K + i * size,
                                                        V + i * size,
                                                        batch_size,
                                                        seq_len,
                                                        head_num,
                                                        size_per_head);
        }
    }
    //}

    // sync_check_cuda_error();
}

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}
__global__ void add_bias_gelu(float* out, const float* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        float val = out[id];
        if (bias != nullptr) {
            float reg_bias = __ldg(&bias[id % n]);
            val = val + reg_bias;
        }
        out[id] = (float)(gelu(val));
    }
}
__global__ void add_bias_relu(float* out, const float* __restrict bias, int m, int n, int num_models, int* model_offset)
{
    int inter_size = n / num_models;
    // for(int i = 0; i < num_models; i++){
    //  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * inter_size; id += blockDim.x * gridDim.x){
    //      float val = out[id];
    //      if (bias != nullptr) {
    //          val = val + ldg(&bias[id % inter_size + inter_size*i]);
    //      }
    //      out[id] = val > (float)0.0f ? val : (float)0.0f;
    //  }
    // }
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        for (int j = 0; j < num_models; j++) {
            if (id <= m * n / (num_models - j)) {
                float val = out[id];
                if (bias != nullptr) {
                    val = val + ldg(&bias[id % n + n * j]);
                }
                out[id] = val > (float)0.0f ? val : (float)0.0f;
                break;
            }
        }
    }
    // for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
    //     float val = out[id];
    //     if (bias != nullptr) {
    //         int bias_id = id % inter_size + inter_size*(id/inter_size);
    //         float reg_bias = __ldg(&bias[bias_id]);
    //         val = val + reg_bias;
    //     }
    //     out[id] = (float)(gelu(val));
    // }
}
void invokeAddBiasRelu(
    float* out, const float* bias, const int m, const int n, int num_models, int* model_offset, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(float);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        // grid.x = ceil(m * n / 1024.);
        grid.x = 2048;
    }
    //printf("%d %d\n", block.x, grid.x);
    add_bias_relu<<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor, num_models, model_offset);
}
void invokeAddBiasGelu(
    float* out, const float* bias, const int m, const int n, int num_models, cudaStream_t* stream)
{

    int token_num = n / num_models;
    const int data_type_factor = 4 / sizeof(float);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (token_num / 4 / data_type_factor <= 1024) {
        block.x = token_num / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        // grid.x = ceil(m * n / 1024.);
        grid.x = 2048;
    }
    for(int i = 0; i < num_models;i++){
        add_bias_gelu<<<grid, block, 0, stream[i]>>>(out+ i * token_num * m, bias + i * token_num, m, token_num / data_type_factor);
    }
}
template<typename T>
__global__ void generalAddBiasResidualLayerNorm(const T* __restrict input,
                                                const T* __restrict gamma,
                                                const T* __restrict beta,
                                                const T* __restrict bias,
                                                T* output,
                                                T* norm_output,
                                                int m,
                                                int n)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = (float)(ldg(&input[blockIdx.x * n + i]));
        local_out += (float)(output[blockIdx.x * n + i]);
        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        norm_output[blockIdx.x * n + i] =
            (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}
template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T* output,
                                              T* norm_output,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* bias,
                                              int m,
                                              int n,
                                              int num_models,
                                              cudaStream_t* stream)
{
    m = m /num_models;
    int size = m * n;
    dim3 grid(m);
    if (m > 1024) {
        dim3 grid(1024);
    }
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (n % 32 != 0) {
        block.x = 1024;
    }  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    for (int i = 0; i < num_models; i++) {
        int model_size = i * size;
        generalAddBiasResidualLayerNorm<T>
            <<<grid, block, 0, stream[i]>>>(input + model_size, gamma+ i*n, beta+ i*n, bias+ i*n, output+ model_size, norm_output+ model_size, m, n);  // For gpt-3
    }
    
}
template void invokeGeneralAddBiasResidualPreLayerNorm(float* output,
                                              float* norm_output,
                                              const float* input,
                                              const float* gamma,
                                              const float* beta,
                                              const float* bias,
                                              int m,
                                              int n,
                                              int num_models,
                                              cudaStream_t* stream);
}  // namespace fastertransformer