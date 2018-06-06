#ifndef CAFFE_UNIFORM_LOSS_LAYER_HPP_
#define CAFFE_UNIFORM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe
{
   template <typename Dtype>
   class UniformLossLayer : public LossLayer<Dtype>
   {
   public:
      explicit UniformLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
      virtual inline const char* type() const { return "UniformLoss"; }
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
   protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
         const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);                
      virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
         const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);                
      virtual inline int ExactNumBottomBlobs() const { return 1; }
      int STEP_SIZE_; //compute loss and gradient per STEP_SIZE_ iterations 
      int iter_; //current iteration
      int batch_size_;
      int code_length_;
      Dtype alpha_; //desired active rate
      Blob<Dtype> current_code_; //current mean code
      Blob<Dtype> vec_sum_; //vector for sum codes
      Blob<Dtype> temp_code_;
      Blob<Dtype> current_diff_; //current gradient
      Dtype loss_; //current loss
   };
}  // namespace caffe

#endif  // CAFFE_UNIFORM_LOSS_LAYER_HPP_
