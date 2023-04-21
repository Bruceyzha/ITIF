/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template< typename T>
void invokeGeneralLayerNorm_modified(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            int num_models,
                            int* model_offset,
                            cudaStream_t stream);

template<typename T>
void invokeAddBiasResidualLayerNorm_modified(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n,int num_models,int* model_offset,cudaStream_t stream);
template<typename T>
void invokeAddBiasResidual_modified(T* output, const T* input, const T* bias, const int m, const int n, int num_models,int* model_offset,cudaStream_t stream);

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
                               cudaStream_t stream);
void invokeAddBiasRelu(float* out, const float* bias, const int m, const int n, int num_models,int* model_offset ,cudaStream_t stream);
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
                            cudaStream_t* stream);

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
                               cudaStream_t* stream);

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
                                              cudaStream_t* stream);
void invokeAddBiasGelu(
    float* out, const float* bias, const int m, const int n, int num_models, cudaStream_t* stream);
    template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, int num_models, cudaStream_t* stream);

}  // namespace fastertransformer