#include <algorithm>
#include <vector>
#include "omp.h"
#include <iostream>
using namespace std;

#include "caffe/layers/set_loss2_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void SetLoss2Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		cat_per_iter_ = -1;
		im_per_cat_ = -1;
		
		alpha_ = this->layer_param_.set_loss2_param().alpha();
		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);

		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void SetLoss2Layer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)	{	}

	template <typename Dtype>
	void SetLoss2Layer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		// LOG(INFO)<<"set1";

		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* data = bottom[0]->cpu_data();

		// initialization of cat_per_iter_, im_per_cat_ and cat_mean_.

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
			cat_per_iter_ = batch_size_ / j;
			temp_loss_.Reshape(1, 1, 1, code_length_);
			cat_mean_.Reshape(1, 1, cat_per_iter_, code_length_);
			cat_grad_.Reshape(1, 1, cat_per_iter_, code_length_);
		}

		for (int i = 0; i < code_length_; ++i)
		{
			temp_loss_.mutable_cpu_data()[i] = 0;
		}
		// compute cat_mean_
		memset(cat_mean_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_ * cat_per_iter_);
		for ( int i = 0; i < batch_size_; i += im_per_cat_)
		{
			for (int j = 0; j < im_per_cat_; ++j)
			{
				caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + (i + j) * code_length_, 1, cat_mean_.mutable_cpu_data() + int(i / im_per_cat_) * code_length_);
			}
		}

		// compute loss and cat grad
		memset(cat_grad_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_ * cat_per_iter_);
		// float** loss = new float* [cat_per_iter_];
		float loss(0.0);
		// #pragma omp parallel for num_threads(6)		
		for ( int i = 0; i < cat_per_iter_; ++i)
		{
			// loss[i] = new float [cat_per_iter_];
			for (int j = i + 1; j < cat_per_iter_; ++j)
			{				
				caffe_sub<Dtype>(code_length_, cat_mean_.cpu_data() + i * code_length_, cat_mean_.cpu_data() + j * code_length_, temp_loss_.mutable_cpu_data());
				Dtype sub_loss = alpha_ - 0.25 * caffe_cpu_dot<Dtype>(code_length_, temp_loss_.cpu_data(), temp_loss_.cpu_data());
				if (sub_loss > 0)
				{
					// loss[i][j] = sub_loss;
					loss += sub_loss;
					caffe_cpu_axpby<Dtype>(code_length_, 1., temp_loss_.cpu_data(), 1., cat_grad_.mutable_cpu_data() + i * code_length_);
					caffe_cpu_axpby<Dtype>(code_length_, -1., temp_loss_.cpu_data(), 1., cat_grad_.mutable_cpu_data() + j * code_length_);
				}
				// else
				// {
				// 	loss[i][j] = 0;
				// }
			}
		}
		// for (int i = 0; i < 16; ++i)
		// {
		// 	LOG(ERROR)<<data[i*1024];
		// }
		// LOG(ERROR)<<cat_mean_.cpu_data()[0]<<" "<<im_per_cat_;

		// float loss_sum(0.);
		// for (int i = 0; i < cat_per_iter_; ++i)
		// {
		// 	for (int j = i + 1; j < cat_per_iter_; ++j)
		// 	{
		// 		loss_sum += loss[i][j];
		// 	}
		// }
		top[0]->mutable_cpu_data()[0] = 2 * loss / cat_per_iter_ / (cat_per_iter_ - 1);
		// LOG(INFO)<<"set2";

	}


	template <typename Dtype>
	void SetLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		// LOG(INFO)<<"set3";

		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
		}

		Dtype* diff = bottom[0]->mutable_cpu_diff();
		loss_weight_ = top[0]->cpu_diff()[0];

		caffe_scal<Dtype>(code_length_ * cat_per_iter_, - loss_weight_ / cat_per_iter_ / (cat_per_iter_ - 1) / im_per_cat_, cat_grad_.mutable_cpu_data());

		// copy gradient to each sample
		for (int i = 0; i < batch_size_; ++i)
		{
			memcpy(diff + i * code_length_, cat_grad_.cpu_data() + int(i / im_per_cat_) * code_length_, sizeof(Dtype) * code_length_);
		}
		// LOG(FATAL)<<"set4";

	}

	// template <typename Dtype>
	// void SetLoss2Layer<Dtype>::Forward_gpu(
	// 	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	// {
	// 	Forward_cpu(bottom, top);
	// }

	// template <typename Dtype>
	// void SetLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	// 	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	// {
	// 	Backward_cpu(top, propagate_down, bottom);
	// }

	#ifdef CPU_ONLY
	STUB_GPU(SetLoss2Layer);
	#endif

	INSTANTIATE_CLASS(SetLoss2Layer);
	REGISTER_LAYER_CLASS(SetLoss2);
}