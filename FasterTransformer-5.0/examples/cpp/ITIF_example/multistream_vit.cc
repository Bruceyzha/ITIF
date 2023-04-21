#include "src/fastertransformer/models/vit/ViT.h"
#include "stdio.h"
#include "stdlib.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
using namespace fastertransformer;

template<typename T>
struct vit_args {
    ViTTransformer<T>* vit;
    std::vector<Tensor> input_tensors;
    std::vector<Tensor> output_tensors;
     ViTWeight<T>* params;
    cudaStream_t stream;
};
template<typename T>
void* vit_thread(void* args){
    struct vit_args<T>* vit_input = (struct vit_args<T>*)args;
    const int ite = 1;
    for (int i = 0; i < 10; i++) {
        vit_input->vit->forward(&vit_input->output_tensors, &vit_input->input_tensors, vit_input->params);
    }
    CudaTimer cuda_timer(vit_input->stream);
    cuda_timer.start();
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < ite; i++) {
        vit_input->vit->forward(&vit_input->output_tensors, &vit_input->input_tensors, vit_input->params);
    }
    //cudaDeviceSynchronize();
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
int vitExample(int batch_size,
                 int img_size, 
                 int patch_size, 
                 int embed_dim, 
                 int head_num, 
                 int layer_num,
                 int token_classifier);

int main(int argc, char** argv)
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    if (argc != 9) {
        printf(
            "[ERROR] vit_example batch_size img_size patch_size embed_dim head_number layer_num with_cls_token is_fp16\n");
        printf("e.g. ./bin/vit_example 1 224 16 768 12 12 1 0 \n");
        return 0;
    }

    int batch_size = atoi(argv[1]);
    int img_size = atoi(argv[2]);
    int patch_size = atoi(argv[3]);
    int embed_dim = atoi(argv[4]);
    int head_num = atoi(argv[5]);
    int layer_num = atoi(argv[6]);
    int token_classifier = atoi(argv[7]);
    vitExample<float>(batch_size,
                 img_size, 
                 patch_size, 
                 embed_dim, 
                 head_num, 
                 layer_num,
                 token_classifier);
}

template<typename T>
int vitExample(int batch_size,
                 int img_size, 
                 int patch_size, 
                 int embed_dim, 
                 int head_num, 
                 int layer_num,
                 int token_classifier)
{
    int thread_num = 4;
    pthread_t* thread_ids = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    ViTTransformer<T>** vit = new ViTTransformer<T>*[thread_num];
    cudaStream_t stream[thread_num];
    cublasHandle_t cublas_handle[thread_num];
    cublasLtHandle_t cublaslt_handle[thread_num];
    cublasAlgoMap* cublas_algo_map[thread_num];
    Allocator<AllocatorType::CUDA>* allocator[thread_num];
    std::mutex* cublas_wrapper_mutex[thread_num];
    cublasMMWrapper* cublas_wrapper[thread_num];
    cudnnHandle_t cudnn_handle[thread_num];
    const int in_chans = 3;
    const bool with_cls_token = token_classifier > 0;
    const int inter_size = embed_dim * 4;
    const int head_dim = embed_dim / head_num;
    const int seq_len = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    AttentionType attention_type = getAttentionType<T>(head_dim, getSMVersion(), true, seq_len);
    T** out_tensor = new T* [thread_num];
    T** from_tensor = new T* [thread_num];
    struct vit_args<T>* vit_input = new struct vit_args<T>[thread_num];
    for(int i = 0; i < thread_num; i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&cublas_handle[i]);
        cublasLtCreate(&cublaslt_handle[i]);
        cublasSetStream(cublas_handle[i], stream[i]);
        checkCUDNN(cudnnCreate(&cudnn_handle[i]));
        checkCUDNN(cudnnSetStream(cudnn_handle[i], stream[i]));
        cublas_algo_map[i] = new cublasAlgoMap("gemm_config.in", "");
        allocator[i] = new Allocator<AllocatorType::CUDA>(getDevice());
        cublas_wrapper_mutex[i] = new std::mutex();
        cublas_wrapper[i] = new cublasMMWrapper(
            cublas_handle[i], cublaslt_handle[i], stream[i], cublas_algo_map[i], cublas_wrapper_mutex[i], allocator[i]);
        cublas_wrapper[i]->setFP32GemmConfig();
        vit_input[i].stream = stream[i];
        int max_batch = batch_size;
        vit_input[i].vit = new ViTTransformer<T>(max_batch,
                                                   img_size,
                                                   in_chans,
                                                   patch_size,
                                                   embed_dim,
                                                   head_num,
                                                   inter_size,
                                                   layer_num,
                                                   with_cls_token,
                                                   getSMVersion(),
                                                   1.0f,
                                                   stream[i],
                                                   cudnn_handle[i],
                                                   cublas_wrapper[i],
                                                   allocator[i],
                                                   false,
                                                   attention_type);
        vit_input[i].params = new ViTWeight<T>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);
        deviceMalloc(&from_tensor[i], batch_size * img_size * img_size * in_chans, false);
        deviceMalloc(&out_tensor[i], batch_size * seq_len * embed_dim, false);
       
        vit_input[i].input_tensors = std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                            getTensorType<T>(),
                            std::vector<size_t>{(size_t)batch_size, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
                            from_tensor[i]}};
        vit_input[i].output_tensors = std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim},
                                   out_tensor[i]}};
    }

    for (int i = 0; i < thread_num; i++) {
        // Create threads with given worker function and argument
        if (pthread_create(&thread_ids[i], NULL, vit_thread<T>, (void*)&vit_input[i]) != 0) {
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

    // warmup
    for (int i = 0; i < 10; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }

    // profile time
    const int ite = 12;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time ,
                ite);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
//     delete cublas_algo_map;
//     delete cublas_wrapper_mutex;
    return 0;
}