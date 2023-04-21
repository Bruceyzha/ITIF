#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/mpi_utils.h"
// #include "/usr/local/mpich-4.1.1/include/mpi.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

using namespace fastertransformer;

template<typename T>
struct bert_args {
    Bert<T>* bert;
    std::vector<Tensor> input_tensors;
    std::vector<Tensor> output_tensors;
    BertWeight<T>* bert_weights;
    cudaStream_t stream;
};

// template<typename T>
// void bert_process(int rank, int size, bert_args<T>& bert_input)
// {
//     const int ite = 1;
//     CudaTimer cuda_timer(bert_input.stream);
//     cuda_timer.start();
//     struct timeval start_time, end_time;
//     gettimeofday(&start_time, 0);
//     for (int i = 0; i < ite; i++) {
//         bert_input.bert->forward(&bert_input.output_tensors, &bert_input.input_tensors, bert_input.bert_weights);
//     }
//     // cudaDeviceSynchronize();
//     float total_time = cuda_timer.stop();
//     gettimeofday(&end_time, 0);
//     FT_LOG_INFO("Rank %d "
//                 "FT-CPP-time %.2f ms (%d iterations) ",
//                 rank,
//                 total_time,
//                 ite);
//     long seconds = end_time.tv_sec - start_time.tv_sec;
//     long microseconds = end_time.tv_usec - start_time.tv_usec;
//     double start = start_time.tv_sec + 1e-6 * start_time.tv_usec;
//     double end = end_time.tv_sec + 1e-6 * end_time.tv_usec;
//     double elapsed = seconds + 1e-6 * microseconds;
//     printf("Rank %d start %f, end %f, it took %f seconds to complete.\n\n", rank, start, end, elapsed);
// }

template<typename T>
int bertExample(int rank,
                size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool is_remove_padding,
                bool allow_gemm_test = true);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 8 && argc != 9) {
        if (rank == 0) {
            FT_LOG_ERROR("bert_example batch_size num_layers seq_len head_num size_per_head is_fp16 is_remove_padding");
            FT_LOG_ERROR("e.g., ./bin/mps_bert 32 12 32 12 64 0 0");
        }
        MPI_Finalize();
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
        printf("hello world\n");
        // bertExample<float>(
        //     rank,batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding, allow_gemm_test);
    }
    else if (atoi(argv[6]) == 1) {
        bertExample<half>(rank,batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding, allow_gemm_test);
    }
    else {
        if (rank == 0) {
            throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                                 "or 1 (use half). \n "));
        }
        MPI_Finalize();
        return 0;
    }
}   
template<typename T>
int bertExample(int rank,   
                size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool is_remove_padding,
                bool allow_gemm_test)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;
    Bert<T>* bert = nullptr;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasAlgoMap* cublas_algo_map;
    Allocator<AllocatorType::CUDA>* allocator;
    std::mutex* cublas_wrapper_mutex;
    cublasMMWrapper* cublas_wrapper;
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
    T* out_tensor = nullptr;
    T* from_tensor = nullptr;
    struct bert_args<T> bert_input;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");
    allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    cublas_wrapper_mutex = new std::mutex();
    cublas_wrapper = new cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, allocator);
    cublas_wrapper->setFP32GemmConfig();
    bert_input.stream = stream;
    bert_input.bert_weights = new BertWeight<T>(hidden_units, inter_size, num_layers);
    bert_input.bert = new Bert<T>(batch_size,
                                    seq_len,
                                    head_num,
                                    size_per_head,
                                    inter_size,
                                    num_layers,
                                    getSMVersion(),
                                    1.0f,
                                    stream,
                                    cublas_wrapper,
                                    allocator,
                                    false,
                                    attention_type,
                                    false,
                                    ActivationType::Gelu,
                                    LayerNormType::pre_layernorm);
    deviceMalloc(&out_tensor, batch_size * seq_len * head_num * size_per_head, false);
    deviceMalloc(&from_tensor, batch_size * seq_len * head_num * size_per_head, false);
    bert_input.input_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                    getTensorType<T>(),
                                    std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                    from_tensor},
                            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}};
    bert_input.output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                    getTensorType<T>(),
                                    std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                    out_tensor}};

    const int ite = 1;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < ite; i++) {
        bert_input.bert->forward(&bert_input.output_tensors, &bert_input.input_tensors, bert_input.bert_weights);
    }
    float total_time = cuda_timer.stop();
    gettimeofday(&end_time, 0);
    double local_start = start_time.tv_sec + 1e-6 * start_time.tv_usec;
    double local_end = end_time.tv_sec + 1e-6 * end_time.tv_usec;
    printf("rank : %d, local_start : %f, local_end : %f\n", rank, local_start, local_end);
    MPI_Barrier(MPI_COMM_WORLD);
    double global_start, global_end;
    MPI_Reduce(&local_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Step 4: Compute the overall forward pass time in the root process (process 0)
    if (rank == 0) {
        double overall_time = global_end - global_start;
        printf("Overall forward pass time: %f seconds\n", overall_time);
    }
    // double start = start_time.tv_sec + 1e-6 * start_time.tv_usec;
    // double end = end_time.tv_sec + 1e-6 * end_time.tv_usec;
    // double elapsed = end - start;
    // printf("start %f, end %f, it took %f seconds to complete.\n\n", start, end, elapsed);

    // FT_LOG_INFO(""
    //             "FT-CPP-time %.2f ms (%d iterations) ",
    //             total_time,
    //             ite);

    delete bert_input.bert_weights;
    delete bert_input.bert;
    cudaFree(d_sequence_lengths);
    cudaFree(from_tensor);
    cudaFree(out_tensor);
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    delete cublas_wrapper;
    delete allocator;

    return 0;
}
