// create by Longhui Wei
// 主要实现了feature l2 normalization
#ifndef CAFFE_NORMALIZATION_L2_LAYER_HPP_
#define CAFFE_NORMALIZATION_L2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    class NormalizationL2Layer : public NeuronLayer<Dtype> {
    public:
        explicit NormalizationL2Layer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const { return "NormalizationL2"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        Blob<Dtype> squared_; //
    };
    
}  // namespace caffe

#endif  // CAFFE_NORMALIZATION_L2_LAYER_HPP_