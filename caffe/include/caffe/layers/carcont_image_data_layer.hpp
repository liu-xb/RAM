#ifndef CAFFE_CARCONT_IMAGE_DATA_LAYER_HPP_
#define CAFFE_CARCONT_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class CarcontImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CarcontImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~CarcontImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CarcontImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 5; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  // shared_ptr<Caffe::RNG> prefetch_rng_;
  // virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::vector<int> > > lines_;
  vector< vector<int> > id_list_;
  vector< vector<int> > col_list_;
  vector< vector<int> > mod_list_;

  int num_all_id_;
  int num_all_col_;
  int num_all_mod_;
  int lines_id_;
  int ni_, nii_, nm_, nim_, nc_, nic_;
  int my_seed_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
