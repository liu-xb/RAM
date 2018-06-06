#include <algorithm>
#include <vector>

#include "caffe/layers/reduce_layer.hpp"

namespace caffe {

// template <typename Dtype>
// void ReduceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   alpha_ = 0;
//   this->blobs_.resize(1);
//   vector(int) weights_shape(1);
//   weights_shape[0] = 1;
//   this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
//   shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
//     this->layer_param_.inner_product_param().bias_filler()));
//   bias_filler->Fill(this->blobs_[0].get());
// }

template <typename Dtype>
void ReduceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LOG(FATAL)<<"ReduceLayer LAYER CANNOT USE CPU!!!!!!!!!!!!\nGPU ONLY!!";
}

template <typename Dtype>
void ReduceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReduceLayer);
#endif

INSTANTIATE_CLASS(ReduceLayer);
REGISTER_LAYER_CLASS(Reduce);

}  // namespace caffe
