#include <algorithm>
#include <vector>

#include "caffe/layers/reduce_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReduceForward(const int count,
  const Dtype alpha, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = in[index] > alpha ? in[index] : 0;
  }
}

template <typename Dtype>
void ReduceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ++iter_;
  alpha_ = this->blobs_[0]->cpu_data()[0];
  if(iter_ == 25)
  {
    LOG(ERROR)<<"reduce alpha : ******* "<<alpha_;
    iter_ = 0;
  }
  // LOG(INFO)<<alpha_;
  // int channels_ = bottom[0]->channels();
  // int batch_size_ = bottom[0]->num();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // const int code_length_ = count / batch_size_;
  ReduceForward<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, alpha_,
    bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReduceBackward(const int count, const Dtype alpha,
  const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    out_diff[index] =
      in_data[index] > alpha ? in_diff[index] : 0;
  }
}

template <typename Dtype>
void ReduceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    ReduceBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, alpha_,
      top_diff, bottom_data, bottom_diff);
    Dtype sum(0);
    for (int i = 0; i < count; ++i)
    {
      if(top[0]->cpu_data()[i]>0)
      {
        ++sum;
      }
    }
    Dtype rho = 
      this->layer_param_.reduce_param().rho();
    sum /= count;
    // Dtype lr = 
    //   this->layer_param_.reduce_param().lr();
    alpha_ += sum * lr_ * ((1.-rho) / (1.-sum) - rho / sum);
    if (iter_ % 249 == 0)
    {
      LOG(ERROR)<<"this sparsity: "<<sum;
    }
    this->blobs_[0]->mutable_cpu_data()[0] = alpha_;
    this->blobs_[0]->mutable_cpu_diff()[0] = 0;
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReduceLayer);


}  // namespace caffe
