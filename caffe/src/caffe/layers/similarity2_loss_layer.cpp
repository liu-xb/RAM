#include <algorithm>
#include <vector>
#include <string>

#include "caffe/layers/similarity2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void Similarity2LossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		alpha_ = this->layer_param_.similarity_loss_param().alpha();
		beta_ = this->layer_param_.similarity_loss_param().beta();
		cat_per_iter_ = this->layer_param_.similarity_loss_param().cat_per_iter();

		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);
		if (batch_size_ < 2)
		{
			LOG(FATAL) << " The batch size for training this net must be at least 2 ! ";
		}

		vector<int> norm_vec_shape(1, batch_size_);
		norm_vec_.Reshape(norm_vec_shape);
		vector<int> norm_mat_shape(2, batch_size_);
		norm_mat_.Reshape(norm_mat_shape);
		dot_product_mat_.Reshape(norm_mat_shape);
		vector<int> temp_diff_vec_shape(1, code_length_);
		temp_diff_vec_.Reshape(temp_diff_vec_shape);
		imagenetsim_ = this->layer_param_.similarity_loss_param().imagenetsim();
		if(imagenetsim_>0)
		{
			sim_mat_ = new float [1000000];
			FILE* fp;
			string path = this->layer_param_.similarity_loss_param().simpath();

			if( (fp = fopen(path.c_str(),"rb")) == 0)
			{
				LOG(FATAL)<<"cannot find imagenetsim";
			}
			int a = fread(sim_mat_, sizeof(float), 1000000,fp);
			LOG(INFO)<<a<<"AD"<<sim_mat_[0]<<sim_mat_[1]<<sim_mat_[1000000];
		}
	}

	template <typename Dtype>
	void Similarity2LossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void Similarity2LossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		LOG(FATAL) << "similarity2 loss layer cannot run with cpu";
		// const int pair_num = batch_size_ * 2;
		// const int num_per_cat = batch_size_ / cat_per_iter_;
		// const Dtype* bottom_data = bottom[0]->cpu_data();
		// const int label_num = bottom[1]->count() / batch_size_;
		// // const int count = bottom[0]->count();
		// const Dtype* label = bottom[1]->cpu_data();
		// const Dtype loss_weight = top[0]->cpu_diff()[0] / (pair_num * 2 / batch_size_);
		// Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		// Dtype loss(0.0);
		// bool if_sim(0);
		
		// sll_cpu_nrm2<Dtype>(code_length_, batch_size_, bottom_data, norm_vec_.mutable_cpu_data());
		// caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, batch_size_ -1, batch_size_ - 1, 1,
		// 	Dtype(1.0), norm_vec_.cpu_data(), norm_vec_.cpu_data() + 1,
		// 	Dtype(0.0), norm_mat_.mutable_cpu_data());
		// caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_ - 1, batch_size_ - 1, code_length_,
		// 	Dtype(1.0), bottom_data, bottom_data + code_length_,
		// 	Dtype(0.0), dot_product_mat_.mutable_cpu_data());
		// caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
		// srand((unsigned int)(time(NULL)));

		// std::map<int, bool> pair_map;
		// for (int num = 0; num < batch_size_; ++num)
		// {
		// 	//for similar pairs
		// 	int i = num;
		// 	int j = i + 1;
		// 	if ((j%num_per_cat) == 0)
		// 	{
		// 		j -= num_per_cat;
		// 	}
		// 	if (j == 0)
		// 	{
		// 		j = i;
		// 		i = 0;
		// 	}
		// 	if (i == batch_size_ - 1)
		// 	{
		// 		i = j;
		// 		j = batch_size_ - 1;
		// 	}
		// 	int locat = i * (batch_size_ - 1) + j - 1;
		// 	Dtype similarity = (dot_product_mat_.cpu_data()[locat])/(norm_mat_.cpu_data()[locat] + 1e-10);
		// 	if ((similarity < 1) && (similarity > 0))
		// 	{
		// 		if (similarity < alpha_)
		// 		{
		// 			loss += (alpha_ - similarity);
		// 			//gradient w.r.t. image i
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0)*(similarity)/(norm_vec_.cpu_data()[i])/(norm_vec_.cpu_data()[i]),
		// 				bottom_data + i * code_length_, (Dtype)0.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, ((Dtype)1.0) / (norm_mat_.cpu_data()[locat]),
		// 				bottom_data + j * code_length_, (Dtype)1.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0 * loss_weight), temp_diff_vec_.cpu_data(), (Dtype)1.0, bottom_diff + i * code_length_);
		// 			//gradient w.r.t image j
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0)*(similarity)/(norm_vec_.cpu_data()[j])/(norm_vec_.cpu_data()[j]),
		// 				bottom_data + j * code_length_, (Dtype)0.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, ((Dtype)1.0) / (norm_mat_.cpu_data()[locat]),
		// 				bottom_data + i * code_length_, (Dtype)1.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0 * loss_weight), temp_diff_vec_.cpu_data(), (Dtype)1.0, bottom_diff + j * code_length_);
		// 		}
		// 	}

		// 	//for dissimilar pairs
		// 	i = num;
		// 	j = rand() % batch_size_;
		// 	if (label_num > 1)
		// 	{
		// 		if_sim = caffe_cpu_dot(label_num, label + i * label_num, label + j * label_num) > 0;
		// 	}
		// 	else
		// 	{
		// 		if_sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[j])));
		// 	}
		// 	while((if_sim)||(pair_map[i * batch_size_ + j]))
		// 	{
		// 		j = rand() % batch_size_;
		// 		if (label_num > 1)
		// 		{
		// 			if_sim = caffe_cpu_dot(label_num, label + i * label_num, label + j * label_num) > 0;
		// 		}
		// 		else
		// 		{
		// 			if_sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[j])));
		// 		}
		// 	}
		// 	if (j == 0)
		// 	{
		// 		j = i;
		// 		i = 0;
		// 	}
		// 	if (i == batch_size_ - 1)
		// 	{
		// 		i = j;
		// 		j = batch_size_ - 1;
		// 	}
		// 	pair_map[i * batch_size_ + j] = 1;
		// 	pair_map[j * batch_size_ + i] = 1;
		// 	locat = i * (batch_size_ - 1) + j - 1;
		// 	similarity = (dot_product_mat_.cpu_data()[locat])/(norm_mat_.cpu_data()[locat] + 1e-10);
		// 	if ((similarity < 1) && (similarity > 0))
		// 	{
		// 		if (similarity > beta_)
		// 		{
		// 			loss += (similarity - beta_);
		// 			// gradient w.r.t image i
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0)*(similarity)/(norm_vec_.cpu_data()[i])/(norm_vec_.cpu_data()[i]),
		// 				bottom_data + i * code_length_, (Dtype)0.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, ((Dtype)1.0) / (norm_mat_.cpu_data()[locat]),
		// 				bottom_data + j * code_length_, (Dtype)1.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, loss_weight, temp_diff_vec_.cpu_data(), (Dtype)1.0, bottom_diff + i * code_length_);
		// 			//gradient w.r.t image j
		// 			caffe_cpu_axpby<Dtype>(code_length_, (-1.0)*(similarity)/(norm_vec_.cpu_data()[j])/(norm_vec_.cpu_data()[j]),
		// 				bottom_data + j * code_length_, (Dtype)0.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, ((Dtype)1.0) / (norm_mat_.cpu_data()[locat]),
		// 				bottom_data + i * code_length_, (Dtype)1.0, temp_diff_vec_.mutable_cpu_data());
		// 			caffe_cpu_axpby<Dtype>(code_length_, loss_weight, temp_diff_vec_.cpu_data(), (Dtype)1.0, bottom_diff + j * code_length_);
		// 		}
		// 	}
		// }
		// top[0]->mutable_cpu_data()[0] = loss / batch_size_;
	}

	template <typename Dtype>
	void Similarity2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
		}
	}

	#ifdef CPU_ONLY
	STUB_GPU(Similarity2LossLayer);
	#endif

	INSTANTIATE_CLASS(Similarity2LossLayer);
	REGISTER_LAYER_CLASS(Similarity2Loss);
}