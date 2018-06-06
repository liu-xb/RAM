#include <algorithm>
#include <vector>

#include "caffe/layers/binarize1_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Binarize1Forward(const int count,
  const Dtype alpha, const Dtype* in, Dtype* out,
  Dtype maxvalue, Dtype minvalue) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = in[index] > alpha ? maxvalue : minvalue;
  }
}

template <typename Dtype>
void Binarize1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ++iter_;
  alpha_ = this->blobs_[0]->cpu_data()[0];
  if(iter_ == 250)
  {
    LOG(INFO)<<"reduce alpha : ---"<<alpha_;
    iter_ = 0;
  }
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Binarize1Forward<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, alpha_,
      bottom_data, top_data, maxvalue_, minvalue_);
  CUDA_POST_KERNEL_CHECK;
}

// template <typename Dtype>
// __global__ void ReduceBackward(const int count, const Dtype alpha,
//   const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
//   CUDA_KERNEL_LOOP(index, count) {
//     out_diff[index] =
//       in_data[index] > alpha ? in_diff[index] : 0;
//   }
// }

template <typename Dtype>
void Binarize1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    vector<int> temp_diff_shape(1, count);
    temp_diff_.Reshape(temp_diff_shape);
    // ReduceBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
    //   CAFFE_CUDA_NUM_THREADS>>>(count, alpha_,
    //   top_diff, bottom_data, bottom_diff);
    caffe_gpu_memcpy(count * sizeof(Dtype),
      top_diff, bottom_diff);
    // Dtype sum(0);
    // for (int i = 0; i < count; ++i)
    // {
    //   if(top[0]->cpu_data()[i]>0)
    //   {
    //     ++sum;
    //   }
    // }
    // Dtype rho = 
    //   this->layer_param_.reduce_param().rho();
    // sum /= count;
    // Dtype lr = 
    //   this->layer_param_.reduce_param().lr();
    // alpha_ += sum * lr * ((1.-rho) / (1.-sum) - rho / sum);
    // this->blobs_[0]->mutable_cpu_data()[0] = alpha_;
    Dtype temp(0);
    caffe_gpu_asum<Dtype>(count, bottom_diff, &temp);
    caffe_gpu_sub(count, bottom_data, top_data, temp_diff_.mutable_gpu_data());
    caffe_gpu_axpy(count, 1 / count * tradeoff,
      temp_diff_.gpu_data(), bottom_diff);
    this->blobs_[0]->mutable_cpu_diff()[0] = - temp / (Dtype)count;
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(Binarize1Layer);


}  // namespace caffe
