#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/carcont_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CarcontImageDataLayer<Dtype>::~CarcontImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

void my_rand(int num, int min, int max, int* r) //[min, max]
{
  vector<int> a(max-min+1);
  for (int i = 0; i < max-min+1; ++i)
  {
    a[i] = min+i;
  }
  random_shuffle(a.begin(), a.end());
  for (int i = 0; i < num; ++i)
  {
    int locat = i % (max-min+1);
    r[i] = a[locat];
  }
}

void my_rand2(int num, int min, int max, int* r) //[min, max]
{
  for (int i = 0; i < num; ++i)
  {
    r[i] = rand()%(max-min+1) + min;
  }
}

template <typename Dtype>
void CarcontImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  srand( (unsigned)time(NULL) );
  
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  num_all_id_ = this->layer_param_.carcont_param().num_all_id();
  num_all_col_ = this->layer_param_.carcont_param().num_all_col();
  num_all_mod_ = this->layer_param_.carcont_param().num_all_mod();

  ni_ = this->layer_param_.carcont_param().ni(); //number of ids per batch
  nii_ = this->layer_param_.carcont_param().nii(); //number of images of per id
  nc_ = this->layer_param_.carcont_param().nc(); //number of colors per batch
  nic_ = this->layer_param_.carcont_param().nic(); //number of images per color
  nm_ = this->layer_param_.carcont_param().nm(); //number of models per batch
  nim_ = this->layer_param_.carcont_param().nim(); //number of images per models
  id_list_.resize(num_all_id_);
  col_list_.resize(num_all_col_);
  mod_list_.resize(num_all_mod_);

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  // size_t pos;
  std::vector<int> label(3);
  string subline;
  int id, col, mod, num_im = 0;
  while (infile >> subline >> id >> col >> mod) {
    label[0] = id;
    label[1] = col;
    label[2] = mod;
    lines_.push_back(std::make_pair(subline,label));
    id_list_[id].push_back(num_im);
    col_list_[col].push_back(num_im);
    mod_list_[mod].push_back(num_im);
    num_im++;
  }
  CHECK(!id_list_[0].empty()) << "File is empty";
  LOG(INFO) << "A total of " << num_im << " images.";
  for( vector< vector<int> >::iterator it = mod_list_.begin(); it != mod_list_.end(); )
  {
    if (it->empty())
    {
      it = mod_list_.erase(it);
    }
    else 
    {
      ++it;
    }
  }
  for ( vector< vector<int> >::iterator it = col_list_.begin(); it != col_list_.end(); )
  {
    if (it->empty())
    {
      it = col_list_.erase(it);
    }
    else
    {
      ++it;
    }
  }
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0].first, new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0].first;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  const int batch_size = ni_ * nii_ + nc_ * nic_ + nm_ * nim_;//this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape); // image

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape); // id
  top[2]->Reshape(label_shape); // col
  top[3]->Reshape(label_shape); // mod
  top[4]->Reshape(label_shape); // order of each sample in file train/val .txt
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(4, batch_size, 1, 1);
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void CarcontImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = ni_ * nii_ + nc_ * nic_ + nm_ * nim_; //image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  random_shuffle(id_list_.begin(), id_list_.end());
  random_shuffle(col_list_.begin(), col_list_.end());
  random_shuffle(mod_list_.begin(), mod_list_.end());

  int item_id = 0;
  for (int i_id = 0; i_id < ni_; ++i_id) // id
  {
    int* temp_sample_image = new int[nii_];
    int temp_id = i_id%id_list_.size();
    my_rand2(nii_, 0, id_list_[temp_id].size()-1, temp_sample_image);
    for (int i = 0; i < nii_; ++i)
    {
      int temp_locat = id_list_[temp_id][temp_sample_image[i]];
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[temp_locat].first,new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[temp_locat].first;
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      prefetch_label[item_id] = lines_[temp_locat].second[0];             // id
      prefetch_label[batch_size + item_id] = lines_[temp_locat].second[1];// col
      prefetch_label[2 * batch_size + item_id] = lines_[temp_locat].second[2]; // mod
      prefetch_label[3 * batch_size + item_id] = temp_locat; // order of each sample in file train/val .txt
      item_id++;
    }
  }
  for (int i_col = 0; i_col < nc_; ++i_col) // col
  {
    int* temp_sample_image = new int[nic_];
    int temp_col = i_col%col_list_.size();
    my_rand2(nic_, 0, col_list_[temp_col].size()-1, temp_sample_image);
    for (int i = 0; i < nic_; ++i)
    {
      int temp_locat = col_list_[temp_col][temp_sample_image[i]];
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[temp_locat].first, new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[temp_locat].first;
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      prefetch_label[item_id] = lines_[temp_locat].second[0]; // id
      prefetch_label[batch_size + item_id] = lines_[temp_locat].second[1]; // col
      prefetch_label[2 * batch_size + item_id] = lines_[temp_locat].second[2]; // mod
      prefetch_label[3 * batch_size + item_id] = temp_locat; // order of each sample in file train/val .txt
      item_id++;
    }
  }

  for (int i_mod = 0; i_mod < nm_; ++i_mod)
  {
    int* temp_sample_image = new int[nim_];
    int temp_mod = i_mod%mod_list_.size();
    my_rand2(nim_, 0, mod_list_[temp_mod].size()-1,temp_sample_image);
    for (int i = 0; i < nim_; ++i)
    {
      int temp_locat = mod_list_[temp_mod][temp_sample_image[i]];
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[temp_locat].first, new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[temp_locat].first;
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      prefetch_label[item_id] = lines_[temp_locat].second[0]; // id
      prefetch_label[batch_size + item_id] = lines_[temp_locat].second[1]; // col
      prefetch_label[2 * batch_size + item_id] = lines_[temp_locat].second[2]; // mod
      prefetch_label[3 * batch_size + item_id] = temp_locat; // order of each sample in file train/val .txt
      item_id++;
    }
  }
  // batch_timer.Stop();
  // DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  // DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  // DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void CarcontImageDataLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {

    Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
    caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
               top[0]->mutable_cpu_data());
    DLOG(INFO) << "Prefetch copied";
    if (this->output_labels_) {
      caffe_copy(batch->label_.count()/4, batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
      caffe_copy(batch->label_.count()/4, batch->label_.cpu_data()+batch->label_.count()/4,
        top[2]->mutable_cpu_data());
      caffe_copy(batch->label_.count()/4, batch->label_.cpu_data()+batch->label_.count()/2,
        top[3]->mutable_cpu_data());
      caffe_copy(batch->label_.count()/4, batch->label_.cpu_data()+batch->label_.count()/4*3,
        top[4]->mutable_cpu_data());
    }
    this->prefetch_free_.push(batch);
  }

template <typename Dtype>
void CarcontImageDataLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    Forward_cpu(bottom, top);
  }

INSTANTIATE_CLASS(CarcontImageDataLayer);
REGISTER_LAYER_CLASS(CarcontImageData);

}  // namespace caffe
#endif  // USE_OPENCV

// if (this->layer_param_.image_data_param().shuffle()) {
  //   // randomly shuffle data
  //   LOG(INFO) << "Shuffling data";
  //   const unsigned int prefetch_rng_seed = caffe_rng_rand();
  //   prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  //   ShuffleImages();
  // }

// Check if we would need to randomly skip a few data points
  // if (this->layer_param_.image_data_param().rand_skip()) {
  //   unsigned int skip = caffe_rng_rand() %
  //       this->layer_param_.image_data_param().rand_skip();
  //   LOG(INFO) << "Skipping first " << skip << " data points.";
  //   CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
  //   lines_id_ = skip;
  // }

// CHECK_GT(lines_size, lines_id_);
    // cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        // new_height, new_width, is_color);
    // CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // read_time += timer.MicroSeconds();
    // timer.Start();
    // Apply transformations (mirror, crop...) to the image
    // int offset = batch->data_.offset(item_id);
    // this->transformed_data_.set_cpu_data(prefetch_data + offset);
    // this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    // trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    // lines_id_++;
    // if (lines_id_ >= lines_size) {
    //   // We have reached the end. Restart from the first.
    //   DLOG(INFO) << "Restarting data prefetching from start.";
    //   lines_id_ = 0;
    //   if (this->layer_param_.image_data_param().shuffle()) {
    //     ShuffleImages();
    //   }