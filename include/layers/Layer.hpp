#pragma once
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/Shape.hpp"
#include "layers/Tensor.hpp"

class Layer {
 public:
  Layer() = default;
};

template <typename ValueType>
class LayerImpl {
 public:
  LayerImpl() = default;
  LayerImpl(const Shape& inputShape, const Shape& outputShape)
      : inputShape_(inputShape), outputShape_(outputShape) {}
  LayerImpl(const LayerImpl& c) = default;
  LayerImpl& operator=(const LayerImpl& c) = default;
  virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
  Shape get_input_shape() const { return inputShape_; }
  Shape get_output_shape() const { return outputShape_; }
  // weights width x height
  std::pair<Shape, Shape> get_dims() const {
    return std::pair<Shape, Shape>(outputShape_, inputShape_);
  }

 protected:
  Shape inputShape_;
  Shape outputShape_;
};
