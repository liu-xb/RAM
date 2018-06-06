#include <algorithm>
#include <vector>

#include "caffe/layers/p_power1_layer.hpp"

namespace caffe {

template <typename Dtype>
void PPower1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<<"only using cpu will be very very slow";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype p = this->blobs_[0]->cpu_data()[0];
  for(int i = 0;  i < count; ++i)
  {
    top_data[i] = pow(bottom_data[i], p);
  }
  if(iter_ == -1)
  {
    temp_diff_.Reshape(bottom[0]->shape());
  }
  ++iter_;
  if (iter_ == 500)
  {
    LOG(INFO)<<"p = "<<p<<'\n';
    iter_ = 0;
  }
}

template <typename Dtype>
void PPower1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype diff_p(0);
    Dtype p = this->blobs_[0]->cpu_data()[0];
    for (int i = 0; i < count; ++i) 
    {
      bottom_diff[i] = top_diff[i] * p * top_data[i] / bottom_data[i];
      if (bottom_data[i] > 0)
      {
        diff_p += top_diff[i] * top_data[i] * log(bottom_data[i]);
      }
    }
    this->blobs_[0]->mutable_cpu_diff()[0] = diff_p;
  }
}


#ifdef CPU_ONLY
STUB_GPU(PPower1Layer);
#endif

INSTANTIATE_CLASS(PPower1Layer);
REGISTER_LAYER_CLASS(PPower1);

}  // namespace caffe
