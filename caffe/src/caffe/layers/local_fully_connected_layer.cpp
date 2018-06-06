#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_fully_connected_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  //axis = 2; 
  K_ = bottom[0]->count(2);
  C_ = bottom[0]->count(1,2);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = K_ * C_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_ * C_); 
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler())); 
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const int new_K = bottom[0]->count(2);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  CHECK_EQ(4, bottom[0]->num_axes())
      << "Bottom blob of LFClayer should have 4 dimensions";
  M_ = bottom[0]->count(0, 1);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(2);
  top_shape[1] = N_ * C_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
   caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for ( int i = 0; i < M_; ++i){
      for ( int j = 0; j < C_; ++j){
          const Dtype* bottom_data_ij = bottom_data + C_ * K_ * i + K_ * j;
          const Dtype* weight_j  = weight + N_ * K_ * j;
          Dtype* top_data_ij = top_data + N_ * C_ * i + N_ * j;
          caffe_cpu_gemm<Dtype>(
                  CblasNoTrans, CblasNoTrans, 1, N_, K_, (Dtype)1.0,
                  bottom_data_ij, weight_j, (Dtype)0.0, top_data_ij);
      }
  }
  if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * C_, 1,
              (Dtype)1.0, bias_multiplier_.cpu_data(),
              this->blobs_[1]->cpu_data(), (Dtype)1.0, top_data);
  }
}

template <typename Dtype>
void LocalFullyConnectedLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    if (this->param_propagate_down_[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        // Gradient with respect to weight
        for (int i = 0; i < M_;  ++i) {
            for (int j = 0; j < C_; ++j) {
                const Dtype* top_diff_ij = top_diff + N_ * C_ * i + N_ * j;
                const Dtype* bottom_data_ij = 
                        bottom_data + C_ * K_ * i + K_ * j;
                Dtype* weight_diff_j = weight_diff + N_ * K_ * j;
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, 
                        (Dtype)1.0, top_diff_ij, bottom_data_ij, (Dtype)1.0,
                        weight_diff_j);
            }
        }
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
    }
    if (propagate_down[0]) {
        // Gradient with respect to bottom data
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        for(int i = 0; i < M_; ++i) {
            for(int j = 0; j < C_; ++j) {
                const Dtype* top_diff_ij = top_diff + N_ * C_ * i + N_ * j;
                const Dtype* weight_j = weight + N_ * K_ * j;
                Dtype* bottom_diff_ij = bottom_diff + C_ * K_ * i + K_ * j;
                caffe_cpu_gemm<Dtype>(
                        CblasNoTrans, CblasTrans, 1, K_, N_, (Dtype)1.0, 
                        top_diff_ij, weight_j, (Dtype)0.0, bottom_diff_ij);
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(LocalFullyConnectedLayer);
#endif

INSTANTIATE_CLASS(LocalFullyConnectedLayer);
REGISTER_LAYER_CLASS(LocalFullyConnected);

}  // namespace caffe
