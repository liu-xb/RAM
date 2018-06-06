#include <algorithm>
#include <vector>

#include "caffe/layers/uniform_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    void UniformLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        
        STEP_SIZE_ = this->layer_param_.uniform_loss_param().step_size();
        
        alpha_ = this->layer_param_.uniform_loss_param().alpha();
        
        iter_ = -1;
        
        loss_ = 0;
        batch_size_ = bottom[0]->num();
        code_length_  = bottom[0]->count(1);

        vector<int> current_code_shape(1, code_length_);
        current_code_.Reshape(current_code_shape);
        caffe_set(code_length_, Dtype(0.0), current_code_.mutable_cpu_data());
        temp_code_.Reshape(current_code_shape);
        caffe_set(code_length_, Dtype(0.0), temp_code_.mutable_cpu_data());
        
        vector<int> vec_sum_shape(1, batch_size_);
        vec_sum_.Reshape(vec_sum_shape);
        caffe_set(batch_size_, Dtype(1), vec_sum_.mutable_cpu_data());

        std::vector<int> vec_diff_shape(1, batch_size_);
        vec_diff_shape.push_back(code_length_);
        current_diff_.Reshape(vec_diff_shape);
        caffe_set(batch_size_ * code_length_, Dtype(0), current_diff_.mutable_cpu_data());
    }
    
    template <typename Dtype>
    void UniformLossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
        top[0]->Reshape(loss_shape);
    }
    
    template <typename Dtype>
    void UniformLossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const int count = bottom[0]->count();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        for (int i = 0; i < count; ++i)
        {
            bottom_diff[i] = bottom_data[i] > Dtype(0);
        }
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            1, code_length_, batch_size_, (Dtype)1.0, vec_sum_.cpu_data(),
            bottom_diff, (Dtype)0.0, current_code_.mutable_cpu_data());
        
        Dtype mean_value = caffe_cpu_asum<Dtype>(code_length_, current_code_.cpu_data()) / (Dtype)code_length_;
        for (int i = 0; i < code_length_; ++i)
        {
            current_code_.mutable_cpu_data()[i] = (current_code_.cpu_data()[i] - mean_value) * loss_weight;
        }
        
        Dtype loss = caffe_cpu_strided_dot<Dtype>(code_length_, current_code_.cpu_data(), 1,
            current_code_.cpu_data(), 1) / loss_weight / loss_weight / (Dtype)(code_length_ -1);
        
        if (loss > alpha_)
        {
           top[0]->mutable_cpu_data()[0] = loss - alpha_;
           caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, batch_size_, code_length_, 1,
            (Dtype)1.0, vec_sum_.cpu_data(), current_code_.cpu_data(), (Dtype)0.0, bottom_diff);
        }
        else
        {
            caffe_set(count, Dtype(0.0), bottom_diff);
            top[0]->mutable_cpu_data()[0] = 0;
        }
        caffe_set(code_length_, Dtype(0.0), current_code_.mutable_cpu_data());
            // iter_ = 0;
        // }
    }
    template <typename Dtype>
    void UniformLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        if (propagate_down[1])
        {
            LOG(FATAL) << this->type()<< " Layer cannot backpropagate to label inputs.";
        }
    }
    
#ifdef CPU_ONLY
STUB_GPU(UniformLossLayer);
#endif

INSTANTIATE_CLASS(UniformLossLayer);
REGISTER_LAYER_CLASS(UniformLoss);
}  // namespace caffe
