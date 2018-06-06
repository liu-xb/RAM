#ifndef CAFFE_REDUCE_LAYER_HPP_
#define CAFFE_REDUCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include	"caffe/layers/neuron_layer.hpp"
namespace caffe
{
  template <typename Dtype>
class ReduceLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReduceLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {
      	alpha_ = 0;
      	iter_ = -1;
      	if (this->blobs_.size() <= 0)
      	{
      		LOG(INFO)<<'a';
	    	  this->blobs_.resize(1);
				  vector<int> weights_shape(1);
				  weights_shape[0] = 1;
				  this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
				  this->blobs_[0]->mutable_cpu_data()[0] = 0;
      	}
			  this->param_propagate_down_.resize(this->blobs_.size(), true);
        lr_ = this->layer_param_.reduce_param().lr();
      	// STEP_SIZE_ =
       //  	this->layer_param_.reduce_layer_param().setp_size();
      }

  virtual inline const char* type() const { return "Reduce"; }

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
  Dtype alpha_;
  Dtype lr_;
};

}  // namespace caffe

#endif  // CAFFE_REDUCE_LAYER_HPP_
