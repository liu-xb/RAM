#ifndef CAFFE_P_POWER1_LAYER_HPP_
#define CAFFE_P_POWER1_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include	"caffe/layers/neuron_layer.hpp"
namespace caffe
{
  template <typename Dtype>
class PPower1Layer : public NeuronLayer<Dtype> {
 public:
  explicit PPower1Layer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {
      	iter_ = -1;
      	if (this->blobs_.size() <= 0)
      	{
  	      this->blobs_.resize(1);
          vector<int> weights_shape(1);
          weights_shape[0] = 1;
          this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
          this->blobs_[0]->mutable_cpu_data()[0] = this->layer_param_.p_power_param().p();
      	}
			  this->param_propagate_down_.resize(this->blobs_.size(), false);
        LOG(INFO)<<" diff_p is temporarily saved in top_diff, so top_diff is changed.\nnumber of non-zero diff_p is saved in top_data, so top_data is changed."<<"d"<<this->blobs_[0]->cpu_data()[0];
      	// STEP_SIZE_ =
       //  	this->layer_param_.reduce_layer_param().setp_size();
      }

  virtual inline const char* type() const { return "PPower1"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // int STEP_SIZE_;
  int iter_;
  Blob<Dtype> temp_diff_;
};

}  // namespace caffe

#endif  
