#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>
#include <cudnn.h>
#include <arrayfire.h>
#include <iostream>

#define CUDA(call) do {                             \
        cudaError_t e = (call);                     \
        if (e == cudaSuccess) break;                \
        fprintf(stderr, __FILE__": %d: %s (%d)\n",  \
               __LINE__, cudaGetErrorString(e), e); \
        exit(1);                                    \
    } while(0);

#define CUDNN(call) do {                                \
        cudnnStatus_t s = (call);                       \
        if (s == CUDNN_STATUS_SUCCESS) break;           \
        fprintf(stderr, __FILE__": %d: %s (%d)\n",      \
               __LINE__, cudnnGetErrorString(s), s);    \
        exit(1);                                        \
    } while(0);


using af::array;
using af::constant;
using af::dim4;
using af::randu;
using af::sum;
using std::cout;
using std::endl;
using std::vector;

void convBench(::benchmark::State& state, af_dtype type) {
  int w = state.range(0);
  int h = state.range(1);
  int n = state.range(2);

  array img = constant(1.f, w, h, 1, n, type);

  const int batchsize = img.dims(3);
  const int channels  = 1;


  int win_sz = state.range(3);
  array filt = randu(win_sz, win_sz, type);
  cout << img.dims() << endl;
  cout << filt.dims() << endl;

  cudnnHandle_t cudnn;
  CUDNN(cudnnCreate(&cudnn));

  //img = reorder(img, 1, 0, 2, 3); //change order to NxCxHxW
  //img = reorder(img, 3, 1, 0, 2); //change order to NxCxHxW

  cudnnTensorDescriptor_t input_descriptor;
  CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   img.dims(3),
                                   img.dims(2),
                                   img.dims(1),
                                   img.dims(0)));

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   img.dims(3),
                                   img.dims(2),
                                   img.dims(1),
                                   img.dims(0)));

  cudnnFilterDescriptor_t kernel_descriptor;
  CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_NCHW,
                                   1,//in
                                   1,//out
                                   win_sz,//f_h
                                   win_sz));//f_w

  const int pad_height = 2;
  const int pad_width  = 2;
  const int vertical_stride   = 1;
  const int horizontal_stride = 1;
  const int dilation_height   = 1;
  const int dilation_width    = 1;

  cudnnConvolutionDescriptor_t convolution_descriptor;
  CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        pad_height,
                                        pad_width,
                                        vertical_stride,
                                        horizontal_stride,
                                        dilation_height,
                                        dilation_width,
                                        CUDNN_CONVOLUTION,
                                        //CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT
                                       ));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            0,//memory limit
                                            &convolution_algorithm));
  size_t workspace_bytes  = 0;
  convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  //convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  //convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

  CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                               input_descriptor,
                                               kernel_descriptor,
                                               convolution_descriptor,
                                               output_descriptor,
                                               convolution_algorithm,
                                               &workspace_bytes));

  //printf("workspace size: %f MB\n", (workspace_bytes / 1048576.0));

  void* d_workspace;
  CUDA(cudaMalloc(&d_workspace, workspace_bytes));

  // Initialize the kernel array just once
  //float h_kernel[] = {1, 1, 1,
                      //1,-8, 1,
                      //1, 1, 1};

  //array kernel(3, 3, h_kernel, afHost);
  array kernel = constant(1.f / (win_sz * win_sz) , win_sz, win_sz);
  kernel = tile(kernel, 1, 1, 1, 1);
  kernel = kernel.as(f32);
  cout << kernel.dims() << endl;

  //float *d_kernel;
  //CUDA(cudaMalloc(&d_kernel, kernel.elements() * sizeof(float)));
  //CUDA(cudaMemcpy(d_kernel, kernel.device<float>(), kernel.elements() * sizeof(float), cudaMemcpyDeviceToDevice));

  kernel.unlock();

  array output = constant(0.f, img.dims(), f32);

  const float alpha = 1.f;
  const float beta  = 0.f;

  CUDNN(cudnnConvolutionForward(cudnn,
                                &alpha,
                                input_descriptor,
                                img.device<float>(),
                                kernel_descriptor,
                                kernel.device<float>(),
                                convolution_descriptor,
                                convolution_algorithm,
                                d_workspace,
                                workspace_bytes,
                                &beta,
                                output_descriptor,
                                output.device<float>()));
  CUDA(cudaStreamSynchronize(0));
  for(auto _ : state) {
    CUDNN(cudnnConvolutionForward(cudnn,
                                  &alpha,
                                  input_descriptor,
                                  img.device<float>(),
                                  kernel_descriptor,
                                  kernel.device<float>(),
                                  convolution_descriptor,
                                  convolution_algorithm,
                                  d_workspace,
                                  workspace_bytes,
                                  &beta,
                                  output_descriptor,
                                  output.device<float>()));
    CUDA(cudaStreamSynchronize(0));
  }

  img.unlock();
  kernel.unlock();
  output.unlock();
  //af_print(output);
  //output = reorder(output, 1, 0, 2, 3);
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32};

  benchmark::Initialize(&argc, argv);
  af::benchmark::RegisterBenchmark("conv_relu_pool", types, convBench)
    ->RangeMultiplier(4)
    //->Ranges({{8,8}, {8, 8}, {1, 1}, {5, 5}})
    ->Ranges({{32, 1<<10}, {32, 1<<10}, {1, 512}, {5, 5}})
    ->ArgNames({"dim0", "dim1", "batchsize", "dim_filt"});

  af::benchmark::AFReporter r;
  af::benchmark::AFJSONReporter jsr;
  benchmark::RunSpecifiedBenchmarks(&r, &jsr);
}

