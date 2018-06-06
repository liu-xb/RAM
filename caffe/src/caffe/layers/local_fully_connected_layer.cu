#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_fully_connected_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  const Dtype** bottom_i = NULL;
  Dtype** top_i = NULL;
  const Dtype** weight_i = NULL;
  size_t size = C_ * sizeof(Dtype*);
  cudaMalloc(&bottom_i, size);
  cudaMalloc(&top_i, size);
  cudaMalloc(&weight_i, size);

  Dtype* alpha = new Dtype [C_];
  Dtype* beta = new Dtype [C_];
  const Dtype** bottom_host = new const Dtype* [C_];
  Dtype** top_host = new Dtype* [C_];
  const Dtype ** weight_host = new const Dtype* [C_];

  for ( int i = 0; i < C_; ++i) {
	bottom_host [i] = bottom_data + K_ * i;
	top_host [i] = top_data + N_ * i;
	weight_host [i] = weight + N_ * K_ * i;
	alpha [i] = (Dtype)1.0;
	beta [i] = (Dtype)0.0;
  }

  CUDA_CHECK(cudaMemcpy(bottom_i, bottom_host, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(weight_i, weight_host, size, cudaMemcpyHostToDevice));
  cudaMemcpy(top_i, top_host, size, cudaMemcpyHostToDevice);
  lfcl_gpu_gemmBatched_forward(M_, N_, K_, C_, alpha, bottom_i, weight_i, beta, top_i);

  cudaFree(bottom_i);
  cudaFree(top_i);
  cudaFree(weight_i);
  delete alpha;
  delete beta;
  delete bottom_host;
  delete top_host;
  delete weight_host;
  //for ( int i = 0; i < C_; ++i){
  //        const Dtype* bottom_data_i = bottom_data + K_ * i;
  //        const Dtype* weight_i  = weight + N_ * K_ * i;
  //        Dtype* top_data_i = top_data + N_ * i;
  //        lfcl_gpu_gemm<Dtype>(M_, N_, K_, C_, (Dtype)1.0,
  //               bottom_data_i, weight_i, (Dtype)0.0, top_data_i);
  //}

  if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, C_ * N_, 1,
              (Dtype)1.0, bias_multiplier_.gpu_data(),
              this->blobs_[1]->gpu_data(), (Dtype)1.0, top_data);
  }
/*const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for ( int i = 0; i < C_; ++i){
          const Dtype* bottom_data_ij = bottom_data + K_ * i;
          const Dtype* weight_j  = weight + N_ * K_ * i;
          Dtype* top_data_ij = top_data + N_ * i;
          lfcl_cpu_gemm<Dtype>(M_, N_, K_, C_, (Dtype)1.0,
                  bottom_data_ij, weight_j, (Dtype)0.0, top_data_ij);
  }*/
}

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
	
        // Gradient with respect to weight
	const Dtype** bottom_i = NULL;
 	const Dtype** top_i = NULL;
  	Dtype** weight_i = NULL;
  	size_t size = C_ * sizeof(Dtype*);
  	cudaMalloc(&bottom_i, size);
  	cudaMalloc(&top_i, size);
  	cudaMalloc(&weight_i, size);

  	Dtype* alpha = new Dtype [C_];
  	//Dtype* beta = new Dtype [C_];
  	const Dtype** bottom_host = new const Dtype* [C_];
  	const Dtype** top_host = new const Dtype* [C_];
  	Dtype ** weight_host = new Dtype* [C_];

  	for ( int i = 0; i < C_; ++i) {
	    bottom_host [i] = bottom_data + K_ * i;
	    top_host [i] = top_diff + N_ * i;
	    weight_host [i] = weight_diff + N_ * K_ * i;
	    alpha [i] = (Dtype)1.0;
	    //beta [i] = (Dtype)0.0;
  	}

  	cudaMemcpy(bottom_i, bottom_host, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(weight_i, weight_host, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(top_i, top_host, size, cudaMemcpyHostToDevice);
  	lfcl_gpu_gemmBatched_backward_weight(M_, N_, K_, C_, alpha, top_i, bottom_i, alpha, weight_i);

  	cudaFree(bottom_i);
  	cudaFree(top_i);
  	cudaFree(weight_i);
  	delete alpha;
//  	delete beta;
  	delete bottom_host;
  	delete top_host;
  	delete weight_host;
       /* for (int i = 0; i < M_;  ++i) {
            for (int j = 0; j < C_; ++j) {
                const Dtype* top_diff_ij = top_diff + N_ * C_ * i + N_ * j;
                const Dtype* bottom_data_ij = 
                        bottom_data + C_ * K_ * i + K_ * j;
                Dtype* weight_diff_j = weight_diff + N_ * K_ * j;
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, 
                        (Dtype)1.0, top_diff_ij, bottom_data_ij, (Dtype)1.0,
                        weight_diff_j);
            }
        }*/
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
   	const Dtype* top_diff = top[0]->gpu_diff();
   	// Gradient with respect to bias
    	caffe_gpu_gemv<Dtype>(CblasTrans, M_, C_ * N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
    }
    if (propagate_down[0]) {
        // Gradient with respect to bottom data
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* weight = this->blobs_[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//////////
	Dtype** bottom_i = NULL;
 	const Dtype** top_i = NULL;
  	const Dtype** weight_i = NULL;
  	size_t size = C_ * sizeof(Dtype*);
  	cudaMalloc(&bottom_i, size);
  	cudaMalloc(&top_i, size);
  	cudaMalloc(&weight_i, size);

  	Dtype* alpha = new Dtype [C_];
  	Dtype* beta = new Dtype [C_];
  	Dtype** bottom_host = new Dtype* [C_];
  	const Dtype** top_host = new const Dtype* [C_];
  	const Dtype ** weight_host = new const Dtype* [C_];

  	for ( int i = 0; i < C_; ++i) {
	    bottom_host [i] = bottom_diff + K_ * i;
	    top_host [i] = top_diff + N_ * i;
	    weight_host [i] = weight + N_ * K_ * i;
	    alpha [i] = (Dtype)1.0;
	    beta [i] = (Dtype)0.0;
  	}

  	cudaMemcpy(bottom_i, bottom_host, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(weight_i, weight_host, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(top_i, top_host, size, cudaMemcpyHostToDevice);
  	lfcl_gpu_gemmBatched_backward_bottom(M_, N_, K_, C_, alpha, weight_i, top_i, beta, bottom_i);

  	cudaFree(bottom_i);
  	cudaFree(top_i);
  	cudaFree(weight_i);
  	delete alpha;
  	delete beta;
  	delete bottom_host;
  	delete top_host;
  	delete weight_host;
        /*for(int i = 0; i < M_; ++i) {
            for(int j = 0; j < C_; ++j) {
                const Dtype* top_diff_ij = top_diff + N_ * C_ * i + N_ * j;
                const Dtype* weight_j = weight + N_ * K_ * j;
                Dtype* bottom_diff_ij = bottom_diff + C_ * K_ * i + K_ * j;
                caffe_gpu_gemm<Dtype>(
                        CblasNoTrans, CblasTrans, 1, K_, N_, (Dtype)1.0, 
                        top_diff_ij, weight_j, (Dtype)0.0, bottom_diff_ij);
            }
        }*/
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalFullyConnectedLayer);

}  // namespace caffe
