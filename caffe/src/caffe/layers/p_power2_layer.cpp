#include <algorithm>
#include <vector>

#include "caffe/layers/p_power2_layer.hpp"

namespace caffe {

template <typename Dtype>
void PPower2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        if (this->blobs_.size() <= 0)
        {
          this->blobs_.resize(1);
          const int channels =  bottom[0]->shape(1);
          vector<int> weights_shape(1, channels);
          this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
          for (int i = 0; i < channels; ++i)
          {
            this->blobs_[0]->mutable_cpu_data()[i] = this->layer_param_.p_power_param().p();
          }
        }

}
template <typename Dtype>
void PPower2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->shape(0);
  const int channels =  bottom[0]->shape(1);
  const int height = bottom[0]->shape(2);
  const int width = bottom[0]->shape(3);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // const int count = bottom[0]->count();
  const Dtype* p = this->blobs_[0]->cpu_data();
  for(int i = 0;  i < batch_size; ++i)
  {
    for (int j = 0; j < channels; ++j)
    {
      for (int k = 0; k < height * width; ++k)
      {
        int temp = i * channels * height * width + j * height * width + k;
        top_data[temp] = pow(bottom_data[temp], p[j]);
      }
    }
  }
  if (iter_ == -1)
  {
    Dtype temp=0;
    for (int i = 0; i < channels; ++i)
    {
      temp += p[i];
    }
    LOG(INFO)<<"mean(p) = "<<temp/channels<<'\n';
  }
  ++iter_;
  if (iter_ == 500)
  {
    Dtype temp = 0;
    for (int i = 0; i < channels; ++i)
    {
      temp += p[i];
    }
    LOG(INFO)<<"mean(p) = "<<temp/channels<<'\n';
    iter_ = 0;
  }
}

template <typename Dtype>
void PPower2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int batch_size = bottom[0]->shape(0);
    const int channels =  bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);
    caffe_set(channels, Dtype(0.0), this->blobs_[0]->mutable_cpu_diff());
    const Dtype* p = this->blobs_[0]->cpu_data();
    for (int i = 0; i < batch_size; ++i) 

    {
      for (int j = 0; j < channels; ++j)
      {
        for (int k = 0; k < height * width; ++k)
        {
          int temp = i * channels * height * width + j * height * width + k;
          bottom_diff[temp] = top_diff[temp] * p[j] * top_data[temp] / bottom_data[temp];
          if (bottom_data[temp] > 0)
          {
            this->blobs_[0]->mutable_cpu_diff()[j] += top_diff[temp] * top_data[temp] * log(bottom_data[temp]);
          }
        }
      }
      
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PPower2Layer);
#endif

INSTANTIATE_CLASS(PPower2Layer);
REGISTER_LAYER_CLASS(PPower2);

}  // namespace caffe
