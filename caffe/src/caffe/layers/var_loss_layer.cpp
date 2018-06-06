#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

#include "caffe/layers/var_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void VarLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		// cat_per_iter_ = -1;
		im_per_cat_ = -1;
		
		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);

		temp_diff_.Reshape(1, code_length_, 1, 1);
	}

	template <typename Dtype>
	void VarLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)	{
		vector<int> loss_shape(0); // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void VarLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		// LOG(INFO)<<"var1";
		loss_weight_ = top[0]->cpu_diff()[0];
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* data = bottom[0]->cpu_data();
		Dtype* diff = bottom[0]->mutable_cpu_diff();

		// initialization of cat_per_iter_ and im_per_cat_.
		if ( 1 )
		{
			int j;
			for ( j = 0; j < batch_size_; ++j)
			{
				if (label[0] != label[j])
				{
					break;
				}
			}
			im_per_cat_ = j;
			// cat_per_iter_ = batch_size_ / j;
		}

		float loss(0);

		// compute loss and gradient
		for ( int i = 0; i < batch_size_; i += im_per_cat_)
		{
			// compute mean
			Dtype* mean = new Dtype [code_length_];
			for (int j = 0; j < code_length_; ++j)
			{
				mean[j] = 0;
			}
			// caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + i * code_length_, 0., mean);
			for ( int j = 0; j < im_per_cat_; ++j)
			{
				caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + (i + j) * code_length_, 1, mean);
			}

			// compute loss and gradient
			for (int j = 0; j < im_per_cat_; ++j)
			{
				caffe_sub<Dtype>(code_length_, data + (i + j) * code_length_, mean, temp_diff_.mutable_cpu_data());
				loss += caffe_cpu_dot<Dtype>(code_length_, temp_diff_.cpu_data(), temp_diff_.cpu_data());
				caffe_cpu_axpby<Dtype>(code_length_, loss_weight_ / batch_size_, temp_diff_.cpu_data(), 0., diff + (i + j) * code_length_);
			}
		}
		top[0]->mutable_cpu_data()[0] = loss / 2. / batch_size_;
		// LOG(INFO)<<"var2";
	}


	template <typename Dtype>
	void VarLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
		}

		// loss_weight_ = top[0]->cpu_diff()[0];
		// const Dtype* label = bottom[1]->cpu_data();
		// const Dtype* data = bottom[0]->cpu_data();
		// Dtype* diff = bottom[0]->mutable_cpu_diff();

		// // initialization of cat_per_iter_ and im_per_cat_.
		// if ( im_per_cat_ < 0 )
		// {
		// 	int j;
		// 	for ( j = 0; j < batch_size_; ++j)
		// 	{
		// 		if (label[0] != label[j])
		// 		{
		// 			break;
		// 		}
		// 	}
		// 	im_per_cat_ = j;
		// 	// cat_per_iter_ = batch_size_ / j;
		// }

		// float loss(0);

		// // compute loss and gradient
		// for ( int i = 0; i < batch_size_; i += im_per_cat_)
		// {
		// 	// compute mean
		// 	Dtype* mean = new Dtype [code_length_];
		// 	caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + i * code_length_, 0., mean);
		// 	for ( int j = 1; j < im_per_cat_; ++j)
		// 	{
		// 		caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + (i + j) * code_length_, 1, mean);
		// 	}

		// 	// compute loss and gradient
		// 	for (int j = 0; j < im_per_cat_; ++j)
		// 	{
		// 		caffe_sub<Dtype>(code_length_, data + (i + j) * code_length_, mean, temp_diff_.mutable_cpu_data());
		// 		loss += caffe_cpu_dot<Dtype>(code_length_, temp_diff_.cpu_data(), temp_diff_.cpu_data());
		// 		caffe_cpu_axpby<Dtype>(code_length_, loss_weight_ / batch_size_, temp_diff_.cpu_data(), 0., diff + (i + j) * code_length_);
		// 	}
		// }
		// top[0]->mutable_cpu_data()[0] = loss / 2. / batch_size_;

	}

	template <typename Dtype>
	void VarLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		Forward_cpu(bottom, top);
	}

	template <typename Dtype>
	void VarLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Backward_cpu(top, propagate_down, bottom);
	}

	#ifdef CPU_ONLY
	STUB_GPU(VarLossLayer);
	#endif

	INSTANTIATE_CLASS(VarLossLayer);
	REGISTER_LAYER_CLASS(VarLoss);
}