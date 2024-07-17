#include "layers/OutputLayer.hpp"

namespace itlab_2023 {

std::pair<std::vector<std::string>, Tensor> OutputLayer::top_k(
    const Tensor& input, size_t k) const {
  if (input.get_shape().dims() != 1) {
    throw std::invalid_argument("TopK function accepts only 1D vector");
  }
  std::vector<std::string> reslabels;
  Tensor resvector;
  switch (input.get_type()) {
    case Type::kFloat: {
      auto toppair = top_k_vec<float>(*input.as<float>(), labels_, k);
      reslabels = toppair.first;
      resvector =
          make_tensor(toppair.second, input.get_shape(), input.get_bias());
      break;
    }
    case Type::kInt: {
      auto toppair = top_k_vec<int>(*input.as<int>(), labels_, k);
      reslabels = toppair.first;
      resvector = make_tensor(toppair.second, input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  };
  return make_pair(reslabels, resvector);
}

template <typename ValueType>
std::vector<ValueType> softmax(const std::vector<ValueType>& vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Empty vector in softmax");
  }
  ValueType max_elem = *std::max_element(vec.begin(), vec.end());
  std::vector<ValueType> res = vec;
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = std::exp(res[i] - max_elem);  // <= 1
  }
  ValueType sum = std::accumulate(res.begin(), res.end(), ValueType(0));
  for (size_t i = 0; i < res.size(); i++) {
    res[i] /= sum;
  }
  return res;
}

template <typename ValueType>
bool compare_pair(std::pair<std::string, ValueType> a,
                  std::pair<std::string, ValueType> b) {
  return (a.second > b.second);
}

template <typename ValueType>
std::pair<std::vector<std::string>, std::vector<ValueType>> top_k_vec(
    const std::vector<ValueType>& input, const std::vector<std::string>& labels,
    size_t k) {
  if (input.size() != labels.size()) {
    throw std::invalid_argument("Labels size not equal input size");
  }
  if (k > input.size()) {
    throw std::invalid_argument("K cannot be bigger than input size");
  }
  std::vector<std::pair<std::string, ValueType>> sort_buf(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    sort_buf[i] = std::make_pair(labels[i], input[i]);
  }
  std::sort(sort_buf.begin(), sort_buf.end(), compare_pair<ValueType>);
  std::vector<std::string> res_labels(k);
  std::vector<ValueType> res_input(k);
  for (size_t i = 0; i < k; i++) {
    res_labels[i] = sort_buf[i].first;
    res_input[i] = sort_buf[i].second;
  }
  return std::make_pair(res_labels, res_input);
}

}  // namespace itlab_2023
