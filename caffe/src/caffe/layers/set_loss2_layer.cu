// #include <algorithm>
// #include <vector>
// #include "omp.h"
// #include <iostream>
// using namespace std;

// #include "caffe/layers/set_loss2_layer.hpp"
// #include "caffe/util/math_functions.hpp"
// #include "caffe/util/io.hpp"

// namespace caffe
// {
// 	template <typename Dtype>
// 	void SetLoss2Layer<Dtype>::Forward_gpu(
// 		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
// 	{

// 		const Dtype* label = bottom[1]->cpu_data();
// 		const Dtype* data = bottom[0]->cpu_data();
// 		const Dtype* data_gpu = bottom[0]->gpu_data();

// 		// initialization of cat_per_iter_, im_per_cat_ and cat_mean_.

// 		if ( im_per_cat_ < 0 )
// 		{
// 			int j;
// 			for ( j = 0; j < batch_size_; ++j)
// 			{
// 				if (label[0] != label[j])
// 				{
// 					break;
// 				}
// 			}
// 			im_per_cat_ = j;
// 			cat_per_iter_ = batch_size_ / j;
// 			temp_loss_.Reshape(1, 1, cat_per_iter_, code_length_);
// 			cat_mean_.Reshape(1, 1, cat_per_iter_, code_length_);
// 			cat_grad_.Reshape(1, 1, cat_per_iter_, code_length_);
// 		}

// 		// compute cat_mean_
// 		memset(cat_mean_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_ * cat_per_iter_);
// 		for ( int i = 0; i < batch_size_; i += im_per_cat_)
// 		{
// 			for (int j = 0; j < im_per_cat_; ++j)
// 			{
// 				caffe_gpu_axpby<Dtype>(code_length_, 1. / im_per_cat_, data_gpu + (i + j) * code_length_, 1, cat_mean_.mutable_gpu_data() + int(i / im_per_cat_) * code_length_);
// 			}
// 		}

// 		// compute loss and cat grad
// 		memset(cat_grad_.mutable_cpu_data(), 0, sizeof(Dtype) * code_length_ * cat_per_iter_);
// 		float loss(0);// = new float* [cat_per_iter_];
// 		for ( int i = 0; i < cat_per_iter_; ++i)
// 		{
// 			// loss[i] = new float [cat_per_iter_];
// 			for (int j = i + 1; j < cat_per_iter_; ++j)
// 			{				
// 				caffe_gpu_sub<Dtype>(code_length_, cat_mean_.gpu_data() + i * code_length_, cat_mean_.gpu_data() + j * code_length_, temp_loss_.mutable_gpu_data()+i*code_length_);
// 				Dtype sub_loss;
				
// 				caffe_gpu_dot<Dtype>(code_length_, temp_loss_.gpu_data()+i*code_length_, temp_loss_.gpu_data()+i*code_length_, &sub_loss);
// 				sub_loss = alpha_ - 0.25 * sub_loss;
// 				if (sub_loss > 0)
// 				{
// 					loss += sub_loss;
// 					caffe_gpu_axpby<Dtype>(code_length_, 1., temp_loss_.gpu_data()+i*code_length_, 1., cat_grad_.mutable_gpu_data() + i * code_length_);
// 					caffe_gpu_axpby<Dtype>(code_length_, -1., temp_loss_.gpu_data()+i*code_length_, 1., cat_grad_.mutable_gpu_data() + j * code_length_);
// 				}
// 				// else
// 				// {
// 				// 	loss[i][j] = 0;
// 				// }
// 			}
// 		}

// 		// float loss_sum(0.);
// 		// for (int i = 0; i < cat_per_iter_; ++i)
// 		// {
// 		// 	for (int j = i + 1; j < cat_per_iter_; ++j)
// 		// 	{
// 		// 		loss_sum += loss[i][j];
// 		// 	}
// 		// }
// 		top[0]->mutable_cpu_data()[0] = 2 * loss / cat_per_iter_ / (cat_per_iter_ - 1);
// 	}


// 	template <typename Dtype>
// 	void SetLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
// 		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
// 	{
// 		if (propagate_down[1])
// 		{
// 			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
// 		}

// 		Dtype* diff = bottom[0]->mutable_gpu_diff();
// 		loss_weight_ = top[0]->cpu_diff()[0];

// 		caffe_gpu_scal<Dtype>(code_length_ * cat_per_iter_, - loss_weight_ / cat_per_iter_ / (cat_per_iter_ - 1) / im_per_cat_, cat_grad_.mutable_gpu_data());

// 		// copy gradient to each sample
// 		for (int i = 0; i < batch_size_; ++i)
// 		{
// 			cudaMemcpy(diff + i * code_length_, cat_grad_.gpu_data() + int(i / im_per_cat_) * code_length_, sizeof(Dtype) * code_length_, cudaMemcpyDefault);
// 		}
// 	}

//   INSTANTIATE_LAYER_GPU_FUNCS(SetLoss2Layer);

// }