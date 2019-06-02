#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  const ReLUParameter& param = this->layer_param_.relu_param();
  num_ = param.num();
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  if (num_ == 1) {
    for (int i = 0; i < count/2; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
  else if (num_ == 2) {
    for (int i = 0; i < 128; ++i) {
      for (int j = 224*56; j < 224*168; ++j) {
        int m = i*j;
        top_data[m] = std::max(bottom_data[m], Dtype(0))
            + negative_slope * std::min(bottom_data[m], Dtype(0));
      }
    }
  }
  else if (num_ == 3) {
    for (int i = 0; i < count/2; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
  else if (num_ == 4) {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
  else if (num_ == 5) {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
  else if (num_ == 6 || num_ == 7){
    for (int i = 0; i < 4096; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
