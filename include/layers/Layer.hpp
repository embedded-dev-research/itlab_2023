#pragma once
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layers/Shape.hpp"

template <typename ValueType>
class Layer {
 public:
  Layer() = delete;
  Layer(const Shape& inputShape, const Shape& outputShape)
      : inputShape_(inputShape), outputShape_(outputShape) {}
  Layer(const Layer& c) = default;
  Layer& operator=(const Layer& c) = default;
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
