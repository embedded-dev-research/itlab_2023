#pragma once

#include <iostream>
#include <vector>
#include <type_traits>

#include "Layer.hpp"

enum Type {
  INT = 0,
  DOUBLE,
};

template <typename T>
Type GetTypeEnum() {
  if (std::is_same(T, int)::value)
    return INT;
  if (std::is_same(T, int)::value)
    return DOUBLE;
}

class Tensor {
private:
  Shape size_;
  std::vector<uint8_t> values_;
  Type type_;

  std::vector<uint8_t> SetRightTypeValues();

  template <typename T>
  std::vector<T>* as();
public:
  Tensor(const size_t dims_count, Type type) : size_(dims_count) {
    type_ = type;
    SetRightTypeValues();
  };
  Tensor(const std::vector<size_t>& dims, Type type) : size_(dims) {
    type_ = type;
    SetRightTypeValues();
  }
  Tensor(const Shape& sh, Type type) : size_(sh) {
    type_ = type;
    SetRightTypeValues();
  }
  Tensor(const Tensor& t) = default;

  Shape get_size() const { return size_; }

  template <typename T>
  double& operator()(const std::vector<size_t>& coords); // write
  template <typename T>
  double operator()(const std::vector<size_t>& coords) const; // read

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

