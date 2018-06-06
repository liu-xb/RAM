#include <algorithm>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"

#include "caffe/layers/var_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe 
{
	template <typename TypeParam>
	class VarLossLayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;

	protected:
		VarLossLayerTest()
			: blob_bottom_data_(new Blob<Dtype>(256, 128, 1, 1)),
				blob_bottom_label_(new Blob<Dtype>(256, 1, 1, 1)),
				blob_top_loss_(new Blob<Dtype>())
		{
			// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-1.0);
			filler_param.set_max(1.0);

			UniformFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);

			for (int i = 0; i < 256; ++i)
			{
				blob_bottom_label_->mutable_cpu_data()[i] = int(i/8);
			}
			blob_bottom_vec_.push_back(blob_bottom_label_);

			blob_top_vec_.push_back(blob_top_loss_);
		}

		virtual ~VarLossLayerTest () 
		{
			delete blob_bottom_data_;
			delete blob_bottom_label_;
			delete blob_top_loss_;
		}
		Blob<Dtype>* blob_bottom_data_;
		Blob<Dtype>* blob_bottom_label_;
		Blob<Dtype>* blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

TYPED_TEST_CASE(VarLossLayerTest, TestDtypesAndDevices);

// TYPED_TEST(VarLossLayerTest, TestForward)
// {
// 	typedef typename TypeParam::Dtype Dtype;
// 	LayerParameter layer_param;
// 	VarLossLayer<Dtype> layer(layer_param);
// 	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
// 	layer.Forward(thid->blob_bottom_vec_, this->blob_top_vec_);


// }

TYPED_TEST(VarLossLayerTest, TestGradient)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	VarLossLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);

	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}
}

