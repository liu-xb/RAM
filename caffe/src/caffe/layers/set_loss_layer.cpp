#include <algorithm>
#include <vector>


#include "caffe/layers/set_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void SetLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		cat_per_iter_ = -1;
		im_per_cat_ = -1;
		
		forget_ratio_ = this->layer_param_.set_loss_param().forget_ratio();
		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);
		initial_batch_mean_ = false;

		batch_mean_.Reshape(1, code_length_, 1, 1);
		temp_loss_.Reshape(1, code_length_, 1, 1);

		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void SetLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)	{}

	template <typename Dtype>
	void SetLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{

		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* data = bottom[0]->cpu_data();
		Dtype* diff = bottom[0]->mutable_cpu_diff();

		// initialization of loss_weight_, cat_per_iter_, im_per_cat_ and cat_mean_.
		if ( im_per_cat_ < 0 )
		{
			loss_weight_ = top[0]->cpu_diff()[0];
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
			cat_mean_.Reshape(1, 1, cat_per_iter_, code_length_);
			memset(batch_mean_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_);
		}

		float loss(0);

		// compute cat_mean_
		memset(cat_mean_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_ * cat_per_iter_);
		for ( int i = 0; i < batch_size_; i += im_per_cat_)
		{
			for (int j = 0; j < im_per_cat_; ++j)
			{
				caffe_cpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data + (i + j) * code_length_, 1, cat_mean_.mutable_cpu_data() + int(i / im_per_cat_) * code_length_);
			}
		}

		// compute loss and batch mean vector
		caffe_scal<Dtype>(code_length_, 1. - forget_ratio_, batch_mean_.mutable_cpu_data());
		for ( int i = 0; i < cat_per_iter_; ++i)
		{
			for (int j = i + 1; j < cat_per_iter_; ++j)
			{
				caffe_sub<Dtype>(code_length_, cat_mean_.cpu_data() + i * code_length_, cat_mean_.cpu_data() + j * code_length_, temp_loss_.mutable_cpu_data());
				loss += caffe_cpu_dot<Dtype>(code_length_, temp_loss_.cpu_data(), temp_loss_.cpu_data());
			}
			if (initial_batch_mean_)
			{
				caffe_cpu_axpby<Dtype>(code_length_, forget_ratio_ / cat_per_iter_, cat_mean_.cpu_data() + i * code_length_, 1., batch_mean_.mutable_cpu_data());
			}
			else
			{
				caffe_cpu_axpby<Dtype>(code_length_, 1. / cat_per_iter_, cat_mean_.cpu_data() + i * code_length_, 1., batch_mean_.mutable_cpu_data());
			}
		}
		initial_batch_mean_ = true;

		// compute gradient w.r.t category
		#pragma omp parallel for num_threads(12)
		for (int i = 0; i < cat_per_iter_; ++i)
		{
			caffe_cpu_axpby<Dtype>(code_length_, loss_weight_ / im_per_cat_ / (cat_per_iter_ - 1), batch_mean_.cpu_data(), -loss_weight_ / im_per_cat_ / (cat_per_iter_ - 1), cat_mean_.mutable_cpu_data() + i * code_length_);
		}

		// copy gradient to each sample
		for (int i = 0; i < batch_size_; ++i)
		{
			memcpy(diff + i * code_length_, cat_mean_.cpu_data() + int(i / im_per_cat_) * code_length_, sizeof(Dtype) * code_length_);
		}

		top[0]->mutable_cpu_data()[0] = 1 - 0.5 * loss / cat_per_iter_ / (cat_per_iter_ - 1);

	}


	template <typename Dtype>
	void SetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
		}
	}

	template <typename Dtype>
	void SetLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		Forward_cpu(bottom, top);
	}

	template <typename Dtype>
	void SetLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Backward_cpu(top, propagate_down, bottom);
	}

	#ifdef CPU_ONLY
	STUB_GPU(SetLossLayer);
	#endif

	INSTANTIATE_CLASS(SetLossLayer);
	REGISTER_LAYER_CLASS(SetLoss);
}