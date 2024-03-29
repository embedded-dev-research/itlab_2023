#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

TEST(ewlayer, works_with_minus) {
  EWLayerImpl<double> layer({2, 2}, "minus");
  std::vector<double> input = {2.0, 3.9, 0.1, 2.3};
  std::vector<double> converted_input = {-2.0, -3.9, -0.1, -2.3};
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, works_with_sin) {
  EWLayerImpl<double> layer({2, 2}, "sin");
  std::vector<double> input = {2.0, 3.9, 0.1, 2.3};
  std::vector<double> converted_input(4);
  std::transform(input.begin(), input.end(), converted_input.begin(),
                 sin<double>);
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, relu_test) {
  EWLayerImpl<double> layer({2, 2}, "relu");
  std::vector<double> input = {1.0, -1.0, 2.0, -2.0};
  std::vector<double> converted_input = {1.0, 0.0, 2.0, 0.0};
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, tanh_test) {
  EWLayerImpl<double> layer({2, 2}, "tanh");
  std::vector<double> input = {1.0, -1.0, 2.0, -2.0};
  std::vector<double> converted_input(4);
  std::transform(input.begin(), input.end(), converted_input.begin(),
                 tanh<double>);
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_relu_float) {
  EWLayer layer;
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  layer.run(input, output, "relu");
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*output.as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_relu_int) {
  EWLayer layer;
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output = make_tensor<int>({0});
  std::vector<int> converted_input = {1, 0, 2, 0};
  layer.run(input, output, "relu");
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*output.as<int>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_throws_with_invalid_function) {
  EWLayer layer;
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  ASSERT_ANY_THROW(layer.run(input, output, "abra"));
}

TEST(ewlayer, get_layer_name) {
  EXPECT_EQ(EWLayer::get_name(), "Element-wise layer");
}
