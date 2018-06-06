#include <algorithm>
#include <vector>

#include "caffe/layers/sparsity_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    
    template <typename Dtype>
            void SparsityLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        alpha_ = this->layer_param_.sparsity_loss_param().alpha();
    }
    
    template <typename Dtype>
            void SparsityLossLayer<Dtype>::Reshape(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
        top[0]->Reshape(loss_shape);
    }
    
    template <typename Dtype>
            void SparsityLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) 
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const int count = bottom[0]->count();
        Dtype loss(0.0);
        Dtype* temp_diff = bottom[0]->mutable_cpu_diff();
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        for (int i = 0; i < count; ++i)
        {
            temp_diff[i] = bottom_data[i] > Dtype(0) ? loss_weight : 0;
        }
        loss = caffe_cpu_asum<Dtype>(count, temp_diff) / (Dtype)count / loss_weight;
        if (loss > alpha_)
        {
            loss -= alpha_;
        }
        else
        {
            loss = 0;
            caffe_set(count, Dtype(0), temp_diff);
        }
        top[0]->mutable_cpu_data()[0] = loss;
    }
    
    template <typename Dtype>
            void SparsityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom)
            {
                if (propagate_down[1])
                {
                  LOG(FATAL) << this->type()
                    << " Layer cannot backpropagate to label inputs.";
                }
            }
    
#ifdef CPU_ONLY
STUB_GPU(SparsityLossLayer);
#endif

    INSTANTIATE_CLASS(SparsityLossLayer);
    REGISTER_LAYER_CLASS(SparsityLoss);
}  // namespace caffe
