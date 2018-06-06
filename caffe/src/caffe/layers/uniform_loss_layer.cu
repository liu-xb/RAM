#include <algorithm>
#include <vector>

#include "caffe/layers/uniform_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
  template <typename Dtype>
  __global__  void UniformLossForward(const int n, const Dtype* in,
    Dtype* out)
  {
    CUDA_KERNEL_LOOP(index, n)
    {
      out[index] = in[index] > (Dtype)0.0;
    }
  }

  template <typename Dtype>
  __global__  void UniformLossBackward(const int n, const Dtype* in,
    Dtype* out)
  {
    CUDA_KERNEL_LOOP(index, n)
    {
      out[index] = in[index];
    }
  }

  template <typename Dtype>
  void UniformLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const int count = bottom[0]->count();
    if (loss_weight == 0)
    {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      cudaMemset(bottom_diff, 0, sizeof(Dtype) * count);
      top[0]->mutable_cpu_data()[0] = 0;
    }
    else
    {
      ++iter_;
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      UniformLossForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, bottom_diff);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
        code_length_, batch_size_, (Dtype)1.0, vec_sum_.gpu_data(),
        bottom_diff, (Dtype)0.0, temp_code_.mutable_gpu_data());

      caffe_gpu_axpy<Dtype>(code_length_,
        (Dtype)(1.0 / (Dtype)batch_size_ / (Dtype)STEP_SIZE_),
        temp_code_.gpu_data(), current_code_.mutable_gpu_data());
      if (iter_ < STEP_SIZE_)
      {
        top[0]->mutable_cpu_data()[0] = loss_;
        UniformLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, current_diff_.gpu_data(),
            bottom_diff);
      }
      else
      {
        iter_ = 0;
        Dtype loss(0);
        Dtype sum_grad = 0;
        for(int i = 0; i < code_length_; ++i)
        {
          Dtype rho = current_code_.cpu_data()[i];
          rho = rho >= 1 ? 0.999 : rho;
          rho = rho <= 0 ? 1e-20 : rho;
          loss += alpha_ * log(alpha_ / rho) +
            (1 - alpha_) * log((1 - alpha_) / (1 - rho));
          Dtype grad = (1 - alpha_) / (1 - rho) -
            alpha_ / rho;
          sum_grad += grad * grad;
          for (int j = 0; j < batch_size_; ++j)
          {
            bottom[0]->mutable_cpu_diff()[i + j * code_length_] = grad;
          }
        }
        sum_grad = pow(sum_grad, 0.5);
        caffe_gpu_scale<Dtype>(count, loss_weight / sum_grad, bottom_diff,
          current_diff_.mutable_gpu_data());
        UniformLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, current_diff_.gpu_data(),
          bottom_diff);

        cudaMemset(current_code_.mutable_gpu_data(), 0,
          sizeof(Dtype) * code_length_);
        loss_ = loss / code_length_;
        top[0]->mutable_cpu_data()[0] = loss_;
      }
    }
  }

  template <typename Dtype>
  void UniformLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
  {
    if (propagate_down[1])
      {
        LOG(FATAL) << this->type()
             << " Layer cannot backpropagate to label inputs.";
      }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(UniformLossLayer);

}  // namespace caffe