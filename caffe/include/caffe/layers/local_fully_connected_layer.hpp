#ifndef CAFFE_LOCAL_FULLY_CONNECTED_LAYER_HPP_
#define CAFFE_LOCAL_FULLY_CONNECTED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {

/**
 * @brief Local fully connected layer is proposed in
 *        ""
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LocalFullyConnectedLayer : public InnerProductLayer<Dtype> {
 public:
  explicit LocalFullyConnectedLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LocalFullyConnected"; }
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

  int M_; // batch size
  int K_; // number of input for each subnode
  int N_; // number of output for each subnode
  int C_; // number of channels in bottom blob
  bool bias_term_; // whether have bias term
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_LOCAL_FULLY_CONNECTED_LAYER_HPP_
