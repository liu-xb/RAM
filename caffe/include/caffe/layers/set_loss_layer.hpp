#ifndef CAFFE_SET_LOSS_LAYER_HPP_
#define CAFFE_SET_LOSS_LAYER_HPP_

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
	class SetLossLayer : public LossLayer<Dtype>
	{
	public:
		explicit SetLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "SetLoss"; }
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
		// Dtype alpha_;
		// Dtype beta_;
		int cat_per_iter_;
		int im_per_cat_;
		Blob<Dtype> batch_mean_;
		Blob<Dtype> temp_loss_;
		Blob<Dtype> cat_mean_;
		Dtype loss_weight_;
		Dtype forget_ratio_;
		bool initial_batch_mean_;
		// Blob<Dtype> norm_vec_;
		// Blob<Dtype> norm_mat_;
		// Blob<Dtype> dot_product_mat_;
		// Blob<Dtype> temp_diff_vec_;
	};

}  // namespace caffe

#endif  // CAFFE_SET_LOSS_LAYER_HPP_
