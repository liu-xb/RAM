#ifndef CAFFE_BINARIZE1_LAYER_HPP_
#define CAFFE_BINARIZE1_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include	"caffe/layers/neuron_layer.hpp"
namespace caffe
{
  /*
  parameter alpha is a number in binarize1 layer
  */
  template <typename Dtype>
class Binarize1Layer : public NeuronLayer<Dtype> {
 public:
  explicit Binarize1Layer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {
      	alpha_ = 0;
      	// iter_ = -1;
      	if (this->blobs_.size() <= 0)
      	{
	    	  this->blobs_.resize(1);
				  vector<int> weights_shape(1);
				  weights_shape[0] = 1;
				  this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
				  this->blobs_[0]->mutable_cpu_data()[0] = 0;
      	}
			  this->param_propagate_down_.resize(this->blobs_.size(),
          false);
      	maxvalue_ =
        	this->layer_param_.threshold_param().maxvalue();
        minvalue_ =
          this->layer_param_.threshold_param().minvalue();
        tradeoff =
          this->layer_param_.threshold_param().tradeoff();
      }

  virtual inline const char* type() const { return "Binarize1"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const   vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // int STEP_SIZE_;
  int iter_;
  Dtype alpha_;
  Dtype maxvalue_;
  Dtype minvalue_;
  Blob<Dtype> temp_diff_;
  Dtype tradeoff;
};

}  // namespace caffe

#endif  // CAFFE_BINARIZE_LAYER_HPP_
