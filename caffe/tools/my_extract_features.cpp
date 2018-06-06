#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using std::string;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name"
    "  save_feature_dataset_name  num_mini_batches gpu_id\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }

  int arg_pos;
  uint device_id;

  if (argc == 6) {
    device_id = 0;
  }
  else {
    device_id = atoi(argv[6]);
  }
  LOG(ERROR) << "Using Device_id=" << device_id;
  Caffe::SetDevice(device_id);
  Caffe::set_mode(Caffe::GPU);

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  std::string feature_extraction_proto(argv[++arg_pos]);
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";
  size_t num_features = blob_names.size();

  std::vector<FILE*> fp(num_features, NULL);
  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
        fp[i] = fopen(dataset_names[i].c_str(), "wb");
  }

  // FILE* fp = fopen(argv[++arg_pos], "wb");
  
  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extracting Features";

  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_features; ++i)
    {
      const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);
        fwrite(feature_blob_data, sizeof(Dtype), dim_features, fp[i]);
      }  // for (int n = 0; n < batch_size; ++n)
    }
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  for (int i = 0; i < num_features; i++)
  {
    fclose(fp[i]);
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
