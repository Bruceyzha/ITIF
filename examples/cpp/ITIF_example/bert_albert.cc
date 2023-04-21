#include "src/fastertransformer/layers/itif/encoder.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/ITIF/ITIF.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <list>
using namespace fastertransformer;

int encoderExample(
    size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head, size_t num_models);

int main(int argc, char** argv)
{
    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    int num_models = atoi(argv[6]);
    encoderExample(batch_size, num_layers, seq_len, head_num, size_per_head, num_models);
}

int encoderExample(
    size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head, size_t num_models)
{   
    int min_batch_size = batch_size/4;
    int min_seq_len = seq_len/4;
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;
    int num_streams = 3;
    if(num_models > 3){
        num_streams = num_models;
    }
    cudaStream_t stream[num_streams];
    cublasHandle_t cublas_handle[num_streams];
    cublasLtHandle_t cublaslt_handle[num_streams];
    cublasAlgoMap* cublas_algo_map[num_streams];
    Allocator<AllocatorType::CUDA>* allocator[num_streams];
    std::mutex* cublas_wrapper_mutex[num_streams];
    cublasMMWrapper* cublas_wrapper[num_streams];
    for(int i = 0; i < num_streams; i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&cublas_handle[i]);
        cublasLtCreate(&cublaslt_handle[i]);
        cublasSetStream(cublas_handle[i], stream[i]);
        cublas_algo_map[i] = new cublasAlgoMap("gemm_config.in", "");
        allocator[i] = new Allocator<AllocatorType::CUDA>(getDevice());
        cublas_wrapper_mutex[i] = new std::mutex();
        cublas_wrapper[i] = new cublasMMWrapper(
            cublas_handle[i], cublaslt_handle[i], stream[i], cublas_algo_map[i], cublas_wrapper_mutex[i], allocator[i]);
        cublas_wrapper[i]->setFP32GemmConfig();
    }
    ITIF::itif(min_batch_size,
           min_seq_len,
           batch_size,
           seq_len,
           head_num,
           size_per_head,
           inter_size,
           hidden_units,
           stream[0],
           cublas_wrapper,
           allocator[0]);

    const float **gemm_profiling_results = itif.getGemmProfilingResults();
    const float **batched_gemm_profiling_results = itif.getBatchedGemmProfilingResults();
    float* out_tensor;
    float* from_tensor;
    size_t num_requests = batch_size * num_models;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_seq_len, seq_len);
    std::vector<Node> request_list;
    size_t offset = 0;
    std::list<Node> linked_list;
    linked_list = request_batching(request_list, gemm_profiling_results, batched_gemm_profiling_results, num_models, linked_list);
    size_t size = batch_size * seq_len * head_num * size_per_head;
    deviceMalloc(&out_tensor, num_models*batch_size * seq_len * head_num * size_per_head, false);
    deviceMalloc(&from_tensor,num_models* batch_size * seq_len * head_num * size_per_head, false);
    for (size_t i = 0; i < num_requests; i++) {
        Node node;
        node.tenant_id = i;
        node.length = dis(gen);
        node.size = head_num * size_per_head;
        node.batch_id = 0;
        node.batched_gemm_id = 0;
        node.data = from_tensor + offset;
        offset += node.length * node.size;
        request_list.push_back(node);
    }
    encoder Encoder = encoder(batch_size,
                              seq_len,
                              head_num,
                              size_per_head,
                              inter_size,
                              num_layers,
                              num_models,
                              stream,
                              cublas_wrapper,
                              allocator[0],
                              true,
                              check_array,
                              from_tensor,
                              out_tensor);

    int* h_sequence_lengths = new int[batch_size * num_models];
    // unsigned int seed = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size * num_models, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size * num_models);
    delete[] h_sequence_lengths;
    int* model_offset = new int[num_models];
    for (int i = 0; i < num_models; i++) {
        model_offset[i] = i * size / num_models;
    }
    int* d_model_offset = (int*)allocator[0]->malloc(sizeof(int) * num_models, false);
    cudaMemcpyAsync(d_model_offset, model_offset, sizeof(int) * num_models, cudaMemcpyHostToDevice, stream[0]);
    std::vector<Tensor> input_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<float>(),
                                   std::vector<size_t>{batch_size  * num_models, seq_len, (size_t)(head_num * size_per_head)},
                                   from_tensor},
                            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size * num_models}, d_sequence_lengths}};

    std::vector<Tensor> output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<float>(),
                                   std::vector<size_t>{batch_size * num_models, seq_len, (size_t)(head_num * size_per_head)},
                                   out_tensor}};
    std::vector<Tensor> model_offsets =
        std::vector<Tensor>{Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{num_models}, d_model_offset}};
    float* attention_mask;
    deviceMalloc(&attention_mask,batch_size * seq_len * seq_len* num_models,false);
    const int ite = 12;
    CudaTimer cuda_timer(stream[0]);
    cuda_timer.start();
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    for (int j = 0; j < 1; j++) {
        invokeBuildEncoderAttentionMask(attention_mask, d_sequence_lengths, batch_size, seq_len, stream[0]);
        for (int i = 0; i < num_layers; i++) {
            Encoder.forward(&input_tensors, &output_tensors, &model_offsets,attention_mask,linked_list);
        }
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time/1,
                ite);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    printf("finished\n");
    return 0;
}