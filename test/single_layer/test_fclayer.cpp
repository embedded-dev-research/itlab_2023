#include <vector>

#include "gtest/gtest.h"
#include "layers/FCLayer.hpp"

TEST(fclayer, calculates_correctly1) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  std::vector<double> input = {1, 2};
  std::vector<double> output = layer.run(input);
  EXPECT_NEAR(output[0], 5.5, 1e-5);
  EXPECT_NEAR(output[1], 4.4, 1e-5);
  EXPECT_NEAR(output[2], 12.0, 1e-5);
}

TEST(fclayer, calculates_correctly2) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  std::vector<double> input = {0.5, 0.0};
  std::vector<double> output = layer.run(input);
  EXPECT_NEAR(output[0], 1.5, 1e-5);
  EXPECT_NEAR(output[1], 0.55, 1e-5);
  EXPECT_NEAR(output[2], 1.0, 1e-5);
}

TEST(fclayer, throws_when_greater_input_size) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  std::vector<double> input = {2.0, 1.0, 0.0};
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_less_input_size) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  std::vector<double> input = {2.0};
  ASSERT_ANY_THROW(layer.run(input));
}

TEST(fclayer, throws_when_empty_input) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  std::vector<double> input;
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_empty_weights) {
  const std::vector<double> a1;
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  ASSERT_ANY_THROW(FCLayerImpl<double> layer(a1, wshape, bias));
}
TEST(fclayer, throws_when_empty_bias) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias;
  ASSERT_ANY_THROW(FCLayerImpl<double> layer(a1, wshape, bias));
}

TEST(fclayer, set_get_weight_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  for (size_t i = 0; i < wshape[0]; i++) {
    for (size_t j = 0; j < wshape[1]; j++) {
      EXPECT_NEAR(layer.get_weight(i, j), a1[wshape.get_index({i, j})], 1e-5);
    }
  }
  for (size_t i = 0; i < wshape[0]; i++) {
    for (size_t j = 0; j < wshape[1]; j++) {
      layer.set_weight(i, j, static_cast<double>(i + j));
      EXPECT_NEAR(layer.get_weight(i, j), static_cast<double>(i + j), 1e-5);
    }
  }
}
TEST(fclayer, set_get_bias_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  for (size_t i = 0; i < bias.size(); i++) {
    EXPECT_NEAR(layer.get_bias(i), bias[i], 1e-5);
  }
  for (size_t i = 0; i < bias.size(); i++) {
    layer.set_bias(i, static_cast<double>(i));
    EXPECT_NEAR(layer.get_bias(i), static_cast<double>(i), 1e-5);
  }
}

TEST(fclayer, set_get_weight_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape wshape({3, 3});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  ASSERT_ANY_THROW(layer.get_weight(4, 0));
  ASSERT_ANY_THROW(layer.get_weight(0, 4));
  ASSERT_ANY_THROW(layer.set_weight(4, 0, 1.3));
  ASSERT_ANY_THROW(layer.set_weight(0, 4, 1.3));
}
TEST(fclayer, set_get_bias_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape wshape({3, 3});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  ASSERT_ANY_THROW(layer.get_bias(4));
  ASSERT_ANY_THROW(layer.set_bias(4, 1.3));
}

TEST(fclayer, get_dims_returns_correctly) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  EXPECT_EQ(layer.get_dims().first[0], 3);
  EXPECT_EQ(layer.get_dims().second[0], 2);
}

TEST(fclayer, new_fc_layer_can_run_float) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> a2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output = make_tensor<float>({0});
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  FCLayer layer;
  layer.run(make_tensor<float>({2.0F, 3.0F}), output, weights, bias);
  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_NEAR((*output.as<float>())[i], a2[i], 1e-5);
  }
}

TEST(fclayer, new_fc_layer_can_run_int) {
  const std::vector<int> a1 = {2, 1, 0, 2, 0, 5};
  const std::vector<int> a2 = {7, 6, 16};
  Tensor weights = make_tensor<int>(a1, {3, 2});
  Tensor output = make_tensor<int>({0});
  Shape wshape({3, 2});
  Tensor bias = make_tensor<int>({0, 0, 1});
  FCLayer layer;
  layer.run(make_tensor<int>({2, 3}), output, weights, bias);
  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_NEAR((*output.as<int>())[i], a2[i], 1e-5);
  }
}

TEST(fclayer, new_fc_layer_throws_when_big_input) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output = make_tensor<float>({0});
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  FCLayer layer;
  ASSERT_ANY_THROW(
      layer.run(make_tensor<float>({2.0F, 3.0F, 4.0F}), output, weights, bias));
}

TEST(fclayer, new_fc_layer_throws_with_incorrect_bias_type) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output = make_tensor<float>({0});
  Shape wshape({3, 2});
  Tensor bias = make_tensor<int>({2, 5, 6});
  FCLayer layer;
  ASSERT_ANY_THROW(
      layer.run(make_tensor<float>({2.0F, 3.0F}), output, weights, bias));
}

TEST(fclayer, new_fc_layer_throws_with_incorrect_input_type) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output = make_tensor<float>({0});
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({2, 5, 6});
  FCLayer layer;
  ASSERT_ANY_THROW(layer.run(make_tensor<int>({2, 3}), output, weights, bias));
}

TEST(fclayer, get_layer_name) {
  EXPECT_EQ(FCLayer::get_name(), "Fully-connected layer");
}
