#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/utils/logger.h"
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
using namespace fastertransformer;

template<typename T>
struct bert_args {
    Bert<T>* bert;
    std::vector<Tensor> input_tensors;
    std::vector<Tensor> output_tensors;
    BertWeight<T>* bert_weights;
    cudaStream_t stream;
    size_t thread_batch_size;
};
template<typename T>
void* bert_thread(void* args){
    struct bert_args<T>* bert_input = (struct bert_args<T>*)args;
    const int ite = 1;
    CudaTimer cuda_timer(bert_input->stream);
    cuda_timer.start();
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < ite; i++) {
        bert_input->bert->forward(&bert_input->output_tensors, &bert_input->input_tensors, bert_input->bert_weights);
    }
    cudaStreamSynchronize(bert_input->stream);
    float total_time = cuda_timer.stop();
    gettimeofday(&end_time, 0);
    FT_LOG_INFO(""
                "FT-CPP-time %.2f ms (%d iterations) ",
                total_time ,
                ite);
    long seconds = end_time.tv_sec - start_time.tv_sec;
    long microseconds = end_time.tv_usec - start_time.tv_usec;
    double start = start_time.tv_sec + 1e-6 * start_time.tv_usec;
    double end = end_time.tv_sec + 1e-6 * end_time.tv_usec;
    double elapsed = seconds + 1e-6 * microseconds;
    printf("start %f, end %f, it took %f seconds to complete.\n\n", start, end, elapsed);
    pthread_exit(NULL);

}
template<typename T>
int bertExample(size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool is_remove_padding,
                bool allow_gemm_test = true);

int main(int argc, char** argv)
{
    if (argc != 8 && argc != 9) {
        FT_LOG_ERROR("bert_example batch_size num_layers seq_len head_num size_per_head is_fp16 is_remove_padding");
        FT_LOG_ERROR("e.g., ./bin/bert_example 32 12 32 12 64 0 0");
        return 0;
    }
    bool allow_gemm_test = false;
    if (argc == 9) {
        allow_gemm_test = (atoi(argv[8]) == 1) ? true : false;
    }

    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    bool is_remove_padding = static_cast<bool>(atoi(argv[7]));

    if (atoi(argv[6]) == 0) {
        return bertExample<float>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding, allow_gemm_test);
    }
    else if (atoi(argv[6]) == 1) {
        return bertExample<half>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding, allow_gemm_test);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int bertExample(size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool is_remove_padding,
                bool allow_gemm_test)
{
    int thread_num = 4;
    size_t batch_size_per_thread = batch_size / thread_num;
    size_t remainder = batch_size % thread_num;
    pthread_t* thread_ids = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;
    Bert<T>** bert = new Bert<T>*[thread_num];
    cudaStream_t stream[thread_num];
    cublasHandle_t cublas_handle[thread_num];
    cublasLtHandle_t cublaslt_handle[thread_num];
    cublasAlgoMap* cublas_algo_map[thread_num];
    Allocator<AllocatorType::CUDA>* allocator[thread_num];
    std::mutex* cublas_wrapper_mutex[thread_num];
    cublasMMWrapper* cublas_wrapper[thread_num];
    int* h_sequence_lengths = new int[batch_size];
    unsigned int seed = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
    delete[] h_sequence_lengths;
    AttentionType attention_type = getAttentionType<T>(size_per_head, getSMVersion(), is_remove_padding, seq_len);
    T** out_tensor = new T* [thread_num];
    T** from_tensor = new T* [thread_num];
    struct bert_args<T>* bert_input = new struct bert_args<T>[thread_num];
    for(int i = 0; i < thread_num; i++){
        size_t current_thread_batch_size = batch_size_per_thread + (i < remainder ? 1 : 0);
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
        bert_input[i].stream = stream[i];
        bert_input[i].bert_weights = new BertWeight<T>(hidden_units, inter_size, num_layers);
        bert_input[i].bert = new Bert<T>(current_thread_batch_size,
                                         seq_len,
                                         head_num,
                                         size_per_head,
                                         inter_size,
                                         num_layers,
                                         getSMVersion(),
                                         1.0f,
                                         stream[i],
                                         cublas_wrapper[i],
                                         allocator[i],
                                         false,
                                         attention_type,
                                         false,
                                         ActivationType::Gelu,
                                         LayerNormType::pre_layernorm);
        deviceMalloc(&out_tensor[i], current_thread_batch_size * seq_len * head_num * size_per_head, false);
        deviceMalloc(&from_tensor[i], current_thread_batch_size * seq_len * head_num * size_per_head, false);
        bert_input[i].input_tensors = std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{current_thread_batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   from_tensor[i]},
                            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{current_thread_batch_size}, d_sequence_lengths}};
        bert_input[i].output_tensors = std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{current_thread_batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   out_tensor[i]}};
        bert_input[i].thread_batch_size = current_thread_batch_size;
    }

    for (int i = 0; i < thread_num; i++) {
        // Create threads with given worker function and argument
        if (pthread_create(&thread_ids[i], NULL, bert_thread<T>, (void*)&bert_input[i]) != 0) {
            perror("unable to create thread");
            return 1;
        }
    }
    // Wait for all threads to finish
    for (size_t i = 0; i < thread_num; i++) {
        if (pthread_join(thread_ids[i], NULL) != 0) {
            perror("unable to join thread");
            return 1;
        }
    }



#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}

