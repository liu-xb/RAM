#include <algorithm>
#include <vector>

#include "caffe/layers/sparsity_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SparsityLossForward(const int n, const Dtype* in, Dtype* out, const Dtype value)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    out[index] = in[index] > 0 ? value : 0.0;
  }
}

template <typename Dtype>
__global__ void SparsityLossBackward(const int n, Dtype* out, const Dtype alpha)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    out[index] = out[index] * alpha;
  }
}

template <typename Dtype>
void SparsityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  if (loss_weight == 0)
  {
    top[0]->mutable_cpu_data()[0] = 0;
  }
  else
  {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype loss(0.0);
    SparsityLossForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_diff, loss_weight);
    caffe_gpu_asum(count, bottom_diff, &loss);
    loss /= (Dtype)count * loss_weight;
    if (loss < alpha_) 
    {
      loss = 0;
      cudaMemset(bottom_diff, 0., count);
    }
    else
    {
      loss -= alpha_;
      SparsityLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, loss);
    }
    top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void SparsityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
  {
    if (propagate_down[1])
      {
        LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
      }
  }

INSTANTIATE_LAYER_GPU_FUNCS(SparsityLossLayer);

}  // namespace caffe
