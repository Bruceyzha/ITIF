#include "src/fastertransformer/models/new_vit/new_vit.h"
#include "src/fastertransformer/utils/logger.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
using namespace fastertransformer;

int vit_test(int batch_size,
             int img_size,
             int patch_size,
             int embed_dim,
             int head_num,
             int layer_num,
             int token_classifier,
             size_t num_models);

int main(int argc, char** argv)
{
    // if (argc != 7 ) {
    //     FT_LOG_ERROR("encoder batch_size num_layers seq_len head_num size_per_head ");
    //     FT_LOG_ERROR("e.g., ./bin/bert_example 32 12 32 12 64 0 ");
    //     return 0;
    // }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    if (argc != 9) {
        printf(
            "[ERROR] vit_example batch_size img_size patch_size embed_dim head_number layer_num with_cls_token is_fp16\n");
        printf("e.g. ./bin/vit_example 1 224 16 768 12 12 1 0 \n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int img_size = atoi(argv[2]);
    const int patch_size = atoi(argv[3]);
    const int embed_dim = atoi(argv[4]);
    const int head_num = atoi(argv[5]);
    const int layer_num = atoi(argv[6]);
    const int token_classifier = atoi(argv[7]);
    const int num_models = atoi(argv[8]);
    vit_test(batch_size, img_size, patch_size, embed_dim, head_num, layer_num, token_classifier, num_models);
    return 0;
}

int vit_test(int batch_size,
             int img_size,
             int patch_size,
             int embed_dim,
             int head_num,
             int layer_num,
             int token_classifier,
             size_t num_models)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    int num_streams = 3;
    if (num_models > 3) {
        num_streams = num_models;
    }
    printf("executed\n");
    cudaStream_t stream[num_streams];
    cublasHandle_t cublas_handle[num_streams];
    cublasLtHandle_t cublaslt_handle[num_streams];
    cublasAlgoMap* cublas_algo_map[num_streams];
    printf("executed\n");
    Allocator<AllocatorType::CUDA>* allocator[num_streams];
    std::mutex* cublas_wrapper_mutex[num_streams];
    cublasMMWrapper* cublas_wrapper[num_streams];
    printf("executed %d\n",stream[0]);
    cudnnHandle_t cudnn_handle;
    checkCUDNN(cudnnCreate(&cudnn_handle));
    printf("executed\n");
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
        checkCUDNN(cudnnSetStream(cudnn_handle, stream[0]));
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
    const int in_chans = 3;
    const bool with_cls_token = token_classifier > 0;
    const int inter_size = embed_dim * 4;
    const int head_dim = embed_dim / head_num;
    const int seq_len = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    float* out_tensor;
    float* from_tensor;
    deviceMalloc(&from_tensor, batch_size * img_size * img_size * in_chans * num_models, false);
    deviceMalloc(&out_tensor, batch_size * seq_len * embed_dim * num_models, false);
    int max_batch = batch_size;
    ViTTransformer* vit = new ViTTransformer(max_batch,
                                             img_size,
                                             in_chans,
                                             patch_size,
                                             embed_dim,
                                             head_num,
                                             inter_size,
                                             layer_num,
                                             num_models,
                                             with_cls_token,
                                             getSMVersion(),
                                             1.0f,
                                             stream,
                                             cudnn_handle,
                                             cublas_wrapper,
                                             allocator[0],
                                             from_tensor);

    int* model_offset = new int[num_models];
    for (int i = 0; i < num_models; i++) {
        model_offset[i] = i * batch_size * seq_len * embed_dim * num_models / num_models;
    }
    int* d_model_offset = (int*)allocator[0]->malloc(sizeof(int) * num_models, false);
    cudaMemcpyAsync(d_model_offset, model_offset, sizeof(int) * num_models, cudaMemcpyHostToDevice, stream[0]);
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<float>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
               from_tensor}};

    std::vector<Tensor> output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<float>(),
                                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim},
                                   out_tensor}};
    std::vector<Tensor> model_offsets =
        std::vector<Tensor>{Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{num_models}, d_model_offset}};
    float* attention_mask;
    deviceMalloc(&attention_mask, batch_size * seq_len * seq_len * num_models, false);
    for (int j = 0; j < 10; j++) {
        vit->forward(&output_tensors, &input_tensors, &model_offsets);
    }
    const int ite = 100;
    CudaTimer cuda_timer(stream[0]);
    cuda_timer.start();
    struct timeval start_time, end_time;
    for (int j = 0; j < 1; j++) {
        vit->forward(&output_tensors, &input_tensors, &model_offsets);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                12,
                total_time ,
                ite);

    // delete cublas_algo_map;
    // delete cublas_wrapper_mutex;
    printf("finished\n");
    return 0;
}