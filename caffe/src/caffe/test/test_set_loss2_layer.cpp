#include <algorithm>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"

#include "caffe/layers/set_loss2_layer.hpp"
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
	class SetLoss2LayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;

	protected:
		SetLoss2LayerTest()
			: blob_bottom_data_(new Blob<Dtype>(160, 64, 1, 1)),
				blob_bottom_label_(new Blob<Dtype>(160, 1, 1, 1)),
				blob_top_loss_(new Blob<Dtype>())
		{
			// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-1.0);
			filler_param.set_max(1.0);

			UniformFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);

			for (int i = 0; i < 160; ++i)
			{
				blob_bottom_label_->mutable_cpu_data()[i] = int(i/16);
			}
			blob_bottom_vec_.push_back(blob_bottom_label_);

			blob_top_vec_.push_back(blob_top_loss_);
		}

		virtual ~SetLoss2LayerTest () 
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

TYPED_TEST_CASE(SetLoss2LayerTest, TestDtypesAndDevices);


TYPED_TEST(SetLoss2LayerTest, TestGradient)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	SetLoss2Layer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-3, 2e-4, 1701);

	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}
}

