#ifndef CAFFE_SPARSITY_LOSS_LAYER_HPP_
#define CAFFE_SPARSITY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
            class SparsityLossLayer : public LossLayer<Dtype>
            {
            public:
                explicit SparsityLossLayer(
                        const LayerParameter& param)
                : LossLayer<Dtype>(param) {}
                
                virtual void LayerSetUp(
                        const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
                
                virtual inline const char* type() const { return "SparsityLoss"; }
                
                virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
                
            protected:
                virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
                
                virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
                
                virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                        const vector<bool>& propagate_down, 
                        const vector<Blob<Dtype>*>& bottom);
                
                virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                        const vector<bool>& propagate_down, 
                        const vector<Blob<Dtype>*>& bottom);
                
                virtual inline int ExactNumBottomBlobs() const { return 1; }
                Dtype alpha_;
            };

}  // namespace caffe

#endif  // CAFFE_SPARSITY_LOSS_LAYER_HPP_
