#include <algorithm>
#include <vector>


#include "caffe/layers/simed3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void SimED3LossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		alpha_ = this->layer_param_.similarity_loss_param().alpha();
		beta_ = this->layer_param_.similarity_loss_param().beta();
		// cat_per_iter_ = this->layer_param_.similarity_loss_param().cat_per_iter();

		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);
		if (batch_size_ < 2)
		{
			LOG(FATAL) << " The batch size for training this net "
				<< "must be at least 2 ! ";
		}

		// vector<int> norm_vec_shape(1, batch_size_);
		// norm_vec_.Reshape(norm_vec_shape);
		vector<int> norm_mat_shape(2, batch_size_);
		// norm_mat_.Reshape(norm_mat_shape);
		dot_product_mat_.Reshape(norm_mat_shape);
		vector<int> temp_diff_vec_shape(1, code_length_);
		temp_diff_vec_.Reshape(temp_diff_vec_shape);
		// imagenetsim_ = this->layer_param_.similarity_loss_param().imagenetsim();
		// if(imagenetsim_>0)
		// {
		// 	sim_mat_ = new float [1000000];
		// 	FILE* fp;
		// 	string path = this->layer_param_.similarity_loss_param().simpath();

		// 	if( (fp = fopen(path.c_str(),"rb")) == 0)
		// 	{
		// 		LOG(FATAL)<<"cannot find imagenetsim";
		// 	}
		// 	int a = fread(sim_mat_, sizeof(float), 1000000,fp);
		// 	LOG(INFO)<<a<<"AD"<<sim_mat_[0]<<sim_mat_[1]<<sim_mat_[1000000];
		// }
	}

	template <typename Dtype>
	void SimED3LossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		// Loss layers output a scalar; 0 axes.
		vector<int> loss_shape(0); 
		top[0]->Reshape(loss_shape);
	}

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
	void SimED3LossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{

		// LOG(INFO)<<"s1";
		
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const int label_num = bottom[1]->count() / batch_size_;
		const int count = bottom[0]->count();
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype loss(0.0);
		bool if_sim(0);

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_,
			batch_size_, code_length_, Dtype(1.0), bottom_data,
			bottom_data, Dtype(0.0),
			dot_product_mat_.mutable_cpu_data());

	

		//!!!!bottom diff has to be initialized!!!!!!!!!!!!!!!!!!!!
		 memset(bottom_diff, 0, sizeof(Dtype) * count);

		// srand((unsigned int)(time(NULL)));
		int num_par(0);
		// int temp(0);

		for (int i = 0; i < batch_size_ - 1; ++i)
		{
			for (int j = i + 1; j < batch_size_; ++j)
			{
				// Dtype sub_loss(0.0);
				if_sim = judge(label_num, label, i, j);
				if (!if_sim)
				{
					continue;
				}
				Dtype sim_sam = 
					dot_product_mat_.cpu_data()[i * batch_size_ + j];
				for (int k = 0; k < batch_size_; ++k)
				{
					if_sim = judge(label_num, label, i, k);
					if (if_sim)
					{
						continue;
					}
					++num_par;
					Dtype sim_dissam =
						dot_product_mat_.cpu_data()[i * batch_size_ + k];
					if (alpha_ * 0.5 + sim_dissam - sim_sam < 0)
					{
						continue; //this negative sample is too easy
					}
					if (beta_ * 0.5 + sim_dissam - sim_sam > 0)
					{
						continue; //this negative sample is too hard
					}
					// LOG(INFO)<<"hh";
					loss += alpha_ + 2 * (sim_dissam - sim_sam);
					// temp++;
					// ++num_par;
					//gradient w.r.t. i
					caffe_sub<Dtype>(code_length_,
						bottom[0]->cpu_data() + k * code_length_,
						bottom[0]->cpu_data() + j * code_length_,
						temp_diff_vec_.mutable_cpu_data());
					caffe_cpu_axpby<Dtype>(code_length_,
						loss_weight, temp_diff_vec_.cpu_data(), 1,
						bottom[0]->mutable_cpu_diff() + i * code_length_);
					//gradient w.r.t. j
					caffe_sub<Dtype>(code_length_,
						bottom[0]->cpu_data() + j * code_length_,
						bottom[0]->cpu_data() + i * code_length_,
						temp_diff_vec_.mutable_cpu_data());
					caffe_cpu_axpby<Dtype>(code_length_,
						loss_weight, temp_diff_vec_.cpu_data(), 1,
						bottom[0]->mutable_cpu_diff() + j * code_length_);
					//gradient w.r.t k
					caffe_sub<Dtype>(code_length_,
						bottom[0]->cpu_data() + i * code_length_,
						bottom[0]->cpu_data() + k * code_length_,
						temp_diff_vec_.mutable_cpu_data());
					caffe_cpu_axpby<Dtype>(code_length_,
						loss_weight, temp_diff_vec_.cpu_data(), 1,
						bottom[0]->mutable_cpu_diff() + k * code_length_);
				}
			}
		}
		// LOG(INFO)<<"dadf"<<num_par;
		caffe_scal<Dtype>(count, 1.0 / (num_par + 1), bottom_diff);
					// LOG(INFO)<<"loss"<<loss<<"num:"<<num_par;

		top[0]->mutable_cpu_data()[0] = loss / (num_par + 1e-5);
		// LOG(INFO)<<"s2:  "<<temp;
	}

	template <typename Dtype>
	void SimED3LossLayer<Dtype>::Backward_cpu(
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

	#ifdef CPU_ONLY
	STUB_GPU(SimED3LossLayer);
	#endif

	INSTANTIATE_CLASS(SimED3LossLayer);
	REGISTER_LAYER_CLASS(SimED3Loss);
}