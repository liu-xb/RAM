#ifndef CAFFE_P_POWER2_LAYER_HPP_
#define CAFFE_P_POWER2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include	"caffe/layers/neuron_layer.hpp"
namespace caffe
{
  template <typename Dtype>
class PPower2Layer : public NeuronLayer<Dtype> {
 public:
  explicit PPower2Layer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {
      	iter_ = -1;
			  this->param_propagate_down_.resize(this->blobs_.size(), false);
        // lr_ = this->layer_param_.p_power_param().lr();
      	// STEP_SIZE_ =
       //  	this->layer_param_.reduce_layer_param().setp_size();
      }

  virtual inline const char* type() const { return "PPower2"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top)
  // {
  //   LOG(FATAL)<<"no gpu implementation for ppower1 layer";
  // }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
  // int STEP_SIZE_;
  int iter_;
  // Dtype p_;
  // Dtype lr_;
};

}  // namespace caffe

#endif  
