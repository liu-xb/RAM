#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/p_power1_layer.hpp"

namespace caffe {

template <typename Dtype>
  __global__ void PPower1Forward (const int count, const Dtype* bottom, const Dtype p, Dtype* top)
  {
    CUDA_KERNEL_LOOP(index, count)
    {
      if (bottom[index] > 0)
      {
        top[index] = pow(bottom[index], p);  
      }
    }
  }
template <typename Dtype>
void PPower1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype p = this->blobs_[0]->cpu_data()[0];
  PPower1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, p, top_data);
  CUDA_POST_KERNEL_CHECK;
  if(iter_ == -1)
  {
    temp_diff_.Reshape(bottom[0]->shape());
    LOG(INFO)<<"p = "<<p;
  }
  ++iter_;
  if (iter_ == 500)
  {
    LOG(INFO)<<"p = "<<p<<'\n';
    iter_ = 0;
  }
}

template <typename Dtype>
__global__ void PPOwer1Backward(const int count, Dtype* top_data, const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff, const Dtype p)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    bottom_diff[index] = top_diff[index] * p * top_data[index] / bottom_data[index];
  }
}

template <typename Dtype>
__global__ void PGradient(const int count, Dtype* top_data, const Dtype* bottom_data, const Dtype* top_diff, Dtype* temp_diff)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    if (bottom_data[index] > 0)
    {
      temp_diff[index] = top_diff[index] * top_data[index] * log(bottom_data[index]);
      top_data[index] = 1;
    }
    else
    {
      temp_diff[index] = 0.0;
      top_data[index] = 0.0;
    }
  }
}

template <typename Dtype>
void PPower1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff(); // diff_p is temporarily saved in top_diff, so top_diff is changed.
  Dtype* top_data = top[0]->mutable_gpu_data(); // number of non-zero diff_p is saved in top_data, so top_data is changed.
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* temp_diff = temp_diff_.mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype diff_p(0);
  Dtype diff_p_num(0);
  Dtype p = this->blobs_[0]->cpu_data()[0];
  if (propagate_down[0]) {
    PPOwer1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data, bottom_data, top_diff, bottom_diff, p);
  }
  CUDA_POST_KERNEL_CHECK;
  PGradient<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data, bottom_data, top_diff, temp_diff);
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_dot<Dtype>(count, temp_diff, top_data, &diff_p);
  // caffe_gpu_asum<Dtype>(count, temp_diff, &diff_p);
  // caffe_gpu_asum<Dtype>(count, top_data, &diff_p_num);
  // this->blobs_[0]->mutable_cpu_diff()[0] = diff_p;// / (diff_p_num + 1e-5);
  // LOG(INFO)<<diff_p<<"aa";
  // diff_p = 0;
  // for (int i = 0; i < count; ++i) 
  // {
  //     if (bottom[0]->cpu_data()[i] > 0)
  //     {
  //       Dtype t = top[0]->cpu_diff()[i] * top[0]->cpu_data()[i] * log(bottom[0]->cpu_data()[i]);
  //       diff_p += t;
  //       // LOG(INFO)<<t<<"  adf   "<<temp_diff_.cpu_data()[i];
  //       // ++diff_p_num;
  //     }
  // }
  // LOG(INFO)<<diff_p<<"bb";
  this->blobs_[0]->mutable_cpu_diff()[0] = diff_p; // /diff_p_num;
}
INSTANTIATE_LAYER_GPU_FUNCS(PPower1Layer);
}  // namespace caffe
