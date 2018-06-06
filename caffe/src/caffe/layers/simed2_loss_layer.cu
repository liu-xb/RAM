#include <algorithm>
#include <vector>
#include <map>

#include "caffe/layers/simed2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	template <typename Dtype>
	inline bool judge(int label_num, Dtype* label, int i, int j)
	{
		if (label_num > 1)
		{
			return caffe_cpu_dot(label_num,
				label + i * label_num,
				label + j * label_num) > 0;
		}
		else
		{
			return label[i] == label[j];
		}
	}

	template <typename Dtype>
	void SimED2LossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const int label_num = bottom[1]->count() / batch_size_;
		const int count = bottom[0]->count();
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype loss_weight =	1.0 * top[0]->cpu_diff()[0] /
			batch_size_ / (batch_size_ - 1) * 2;
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype loss(0.0);
		bool if_sim(0);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_,
			batch_size_, code_length_, Dtype(1.0), bottom_data,
			bottom_data, Dtype(0.0),
			dot_product_mat_.mutable_gpu_data());

		// int temp(0);
		// caffe_gpu_max<Dtype>(count, dot_product_mat_.gpu_data(),
		// 	&temp);
		// if (dot_product_mat_.cpu_data()[temp] > 2.0)
		// {
		// 	LOG(FATAL) << "SimED2LossLayer need l2"
		// 		<<" normalization first!";
		// }

		// caffe_gpu_min<Dtype>(count, dot_product_mat_.gpu_data(),
		// 	&temp);
		// if (dot_product_mat_.cpu_data()[temp] < -2.0)
		// {
		// 	LOG(FATAL) << "SimED2LossLayer need l2"
		// 		<<" normalization first!";
		// }

		//!!!!bottom diff must be initialized!!!!!!!!!!
		cudaMemset(bottom_diff, 0, sizeof(Dtype) * count);
		// srand((unsigned int)(time(NULL)));
		// int num_par(0);

		for (int i = 0; i < batch_size_ - 1; ++i)
		{
			for (int j = i + 1; j < batch_size_; ++j)
			{
				Dtype sub_loss(0.0);
				if_sim = judge(label_num, label, i, j);
				if (if_sim)
				{
					sub_loss = 2 - 
						2 * dot_product_mat_.cpu_data()[i * batch_size_ + j];
				}
				else
				{
					sub_loss = alpha_ - 2 + 
						2 * dot_product_mat_.cpu_data()[i * batch_size_ + j];
				}
				if (sub_loss > 0)
				{// whether we need to compute the gradient
					loss += sub_loss;
					int factor = (int)if_sim * 2 - 1;
					caffe_sub<Dtype>(code_length_,
						bottom[0]->cpu_data() + i * code_length_,
						bottom[0]->cpu_data() + j * code_length_,
						temp_diff_vec_.mutable_cpu_data());
					// gradient with respect to i
					caffe_cpu_axpby<Dtype>(code_length_,
						loss_weight * factor, temp_diff_vec_.cpu_data(),
						Dtype(1.0),
						bottom[0]->mutable_cpu_diff() + i * code_length_);
					// gradient with respect to j
					caffe_cpu_axpby<Dtype>(code_length_,
						-loss_weight * factor, temp_diff_vec_.cpu_data(),
						Dtype(1),
						bottom[0]->mutable_cpu_diff() + j * code_length_);
				}
			}
		}
		top[0]->mutable_cpu_data()[0] =
			loss / batch_size_ / (batch_size_ - 1) * 2;
	}

	template <typename Dtype>
	void SimED2LossLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() <<
				" Layer cannot backpropagate to label inputs. ";
		}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(SimED2LossLayer);
}  // namespace caffe


// if (label_num > 1)
				// {
				// 	if_sim = caffe_cpu_dot(label_num, 
				// 		label + i * label_num,
				// 		label + j * label_num) > 0;
				// }
				// else
				// {
				// 	if_sim = ((static_cast<int>(label[i])) == 
				// 		(static_cast<int>(label[j])));
				// }