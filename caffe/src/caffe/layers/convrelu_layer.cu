#include <algorithm>
#include <vector>

#include "caffe/layers/convrelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConvReLUForward(const int channels, const int len, 
  const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, channels) {
    Dtype sum(0);
    for (int i = 0; i < len; ++i)
    {
      sum += in[index * len + i];
    }
    for (int i = 0; i < len; ++i)
    {
      out[i + index * len] = sum > 0 ? in[i + index * len] : 0;
    }
  }
}

template <typename Dtype>
void ConvReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channels_ = bottom[0]->channels();
  int batch_size_ = bottom[0]->num();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int code_length_ = count / batch_size_;
  for (int i = 0; i < batch_size_; ++i)
  {
    ConvReLUForward<Dtype><<<CAFFE_GET_BLOCKS(channels_),
      CAFFE_CUDA_NUM_THREADS>>>(channels_, count / batch_size_ / channels_,
      bottom_data + i * code_length_, top_data + i * code_length_);
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ConvReLUBackward(const int channels, const int len,
  const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, channels) {
    Dtype sum(0);
    for (int i = 0; i < len; ++i)
    {
      sum += in_data[i + index * len];
    }
    for (int i = 0; i < len; ++i)
    {
      out_diff[i + index * len] = sum > 0 ? in_diff[i + index * len] : 0;
    }
  }
}

template <typename Dtype>
void ConvReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    int channels_ = bottom[0]->channels();
    int batch_size_ = bottom[0]->num();
    const int code_length_ = count / batch_size_;
    for (int i = 0; i < batch_size_; ++i)
    {
      ConvReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(channels_),
        CAFFE_CUDA_NUM_THREADS>>>(channels_, count / batch_size_ / channels_,
        top_diff + i * code_length_, bottom_data + i * code_length_,
        bottom_diff + i * code_length_);
    }
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConvReLULayer);


}  // namespace caffe
