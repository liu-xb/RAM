#include <algorithm>
#include <vector>

#include "caffe/layers/global_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void GlobalLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		// alpha_ = this->layer_param_.similarity_loss_param().alpha();
		// beta_ = this->layer_param_.similarity_loss_param().beta();
		// cat_per_iter_ = this->layer_param_.similarity_loss_param().cat_per_iter();

		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);
				vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
		// code_length_  = bottom[0]->count(1);
		// if (batch_size_ < 2)
		// {
		// 	LOG(FATAL) << " The batch size for training this net must be at least 2 ! ";
		// }

	}

	template <typename Dtype>
	void GlobalLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		;
	}

	template <typename Dtype>
	void GlobalLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const	Dtype* sam = bottom[1]->cpu_data();
		const Dtype* sim = bottom[2]->cpu_data();
		const Dtype* gradmultiplier = bottom[3]->cpu_data();
		const Dtype* mappping = bottom[4]->cpu_data();
		const Dtype* mean = bottom[5]->cpu_data();
		const Dtype* bincount = bottom[6]->cpu_data();
		const Dtype loss_weight = top[0]->cpu_diff()[0] * 2.0 / batch_size_ / (batch_size_ - 1);
		const int count = bottom[0]->count();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		cudaMemset(bottom_diff, 0, sizeof(Dtype) * count);

		for (int i = 0; i < batch_size_; ++i)
		{
			for (int j = i + 1; j < batch_size_; ++j)
			{
				int temp = mappping[(int)sam[i * batch_size_ + j]];
				Dtype sub_gradient = (2. * (sim[i * batch_size_ + j] - mean[temp]) + gradmultiplier[temp]) * loss_weight;// / code_length_;

// / bincount[(int)sam[i * batch_size_ + j]]

				caffe_cpu_axpby<Dtype>(code_length_, sub_gradient, bottom_data + j * code_length_, 1., bottom[0]->mutable_cpu_diff() + i * code_length_);

				caffe_cpu_axpby<Dtype>(code_length_, sub_gradient, bottom_data + i * code_length_, 1., bottom[0]->mutable_cpu_diff() + j * code_length_);
			}
		}
		top[0]->mutable_cpu_data()[0] = bottom[7]->cpu_data()[0];
	}

	template <typename Dtype>
	void GlobalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		;
	}

	#ifdef CPU_ONLY
	STUB_GPU(GlobalLossLayer);
	#endif

	INSTANTIATE_CLASS(GlobalLossLayer);
	REGISTER_LAYER_CLASS(GlobalLoss);
}