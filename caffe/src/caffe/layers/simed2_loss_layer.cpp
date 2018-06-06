#include <algorithm>
#include <vector>


#include "caffe/layers/simed2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	template <typename Dtype>
	void SimED2LossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		alpha_ = this->layer_param_.similarity_loss_param().alpha();
		beta_ = this->layer_param_.similarity_loss_param().beta();
		// cat_per_iter_ = this->layer_param_.similarity_loss_param().cat_per_iter();

		batch_size_ = bottom[0]->num();
		code_length_  = bottom[0]->count(1);
		if (batch_size_ < 2)
		{
			LOG(FATAL) << " The batch size for training this net must be at least 2 ! ";
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
	void SimED2LossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void SimED2LossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LOG(FATAL) << "I didn't implement forward with cpu";
	}

	template <typename Dtype>
	void SimED2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1])
		{
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
		}
	}

	#ifdef CPU_ONLY
	STUB_GPU(SimED2LossLayer);
	#endif

	INSTANTIATE_CLASS(SimED2LossLayer);
	REGISTER_LAYER_CLASS(SimED2Loss);
}