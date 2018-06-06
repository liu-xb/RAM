#ifndef CAFFE_VAR_LOSS_LAYER_HPP_
#define CAFFE_VAR_LOSS_LAYER_HPP_

#include <vector>
#include <stdlib.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe
{
	/*
	Compute loss and gradient by Euclidean Distance in pairs.
	loss = 1/2 * 1/n * 
		\sum_n ( E(A, A+) + max(0, alpha - E(A, A-)))
	parameter: alpha
	*/
	template <typename Dtype>
	class VarLossLayer : public LossLayer<Dtype>
	{
	public:
		explicit VarLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "VarLoss"; }
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		int batch_size_;
		int code_length_;
		int cat_per_iter_;
		int im_per_cat_;
		Blob<Dtype> temp_diff_;
		Dtype loss_weight_;
	};

}  // namespace caffe

#endif  // CAFFE_VAR_LOSS_LAYER_HPP_
