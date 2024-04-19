#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

TEST(poolinglayer, empty_inputs1) {
  Shape inpshape = 0;
  Shape poolshape = 0;
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, empty_inputs2) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input;
  ASSERT_ANY_THROW(std::vector<double> output = a.run(input));
}

TEST(poolinglayer, empty_inputs3) {
  Shape inpshape = {3};
  Shape poolshape = {0};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, throws_when_big_input) {
  Shape inpshape = {7};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  ASSERT_ANY_THROW(a.run(input));
}

TEST(poolinglayer, throws_when_invalid_pooling_type) {
  Shape inpshape = {7};
  Shape poolshape = {3};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "my"));
}

TEST(poolinglayer, throws_when_bigger_pooling_dims) {
  Shape inpshape = {8};
  Shape poolshape = {8, 8};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, pooling_throws_when_more_than_2d) {
  Shape inpshape = {4, 4, 4};
  Shape poolshape = {2, 1, 3};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, equivalent_output_when_pool_size_1) {
  Shape inpshape = {8};
  Shape poolshape = {1};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  PoolingLayerImpl<double> b =
      PoolingLayerImpl<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output_a = a.run(input);
  std::vector<double> output_b = b.run(input);
  EXPECT_EQ(output_a, input);
  EXPECT_EQ(output_b, input);
}

class pooling_tests_parameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, Shape, Shape, std::string,
                     std::vector<double> > > {};
// 1) input; 2) input_shape; 3) pooling_type; 4) expected_output;

TEST_P(pooling_tests_parameterized, pooling_works_correctly) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  Shape inpshape = std::get<1>(data);
  Shape poolshape = std::get<2>(data);
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, std::get<3>(data));
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = std::get<4>(data);
  EXPECT_EQ(output, true_output);
}

std::vector<double> basic_1d_data = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0};
Shape basic_1d_shape = {8};

std::vector<double> basic_2d_1_data = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                                       2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
Shape basic_2d_1_shape = {4, 4};

std::vector<double> basic_2d_2_data = {9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0};
Shape basic_2d_2_shape = {3, 3};

INSTANTIATE_TEST_SUITE_P(
    pooling_tests, pooling_tests_parameterized,
    ::testing::Values(
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}),
                        std::string("average"),
                        std::vector<double>({8.0, 5.0, 2.5})),
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}),
                        std::string("max"),
                        std::vector<double>({9.0, 6.0, 3.0})),
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({9}),
                        std::string("average"), std::vector<double>({5.5})),
        std::make_tuple(basic_2d_1_data, basic_2d_1_shape, Shape({2, 2}),
                        std::string("average"),
                        std::vector<double>({6.5, 4.5, 4.5, 6.5})),
        std::make_tuple(basic_2d_1_data, basic_2d_1_shape, Shape({2, 2}),
                        std::string("max"),
                        std::vector<double>({9.0, 7.0, 7.0, 9.0})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({2, 2}),
                        std::string("average"),
                        std::vector<double>({6.5, 5.0, 2.5, 4.0})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({2, 2}),
                        std::string("max"),
                        std::vector<double>({9.0, 7.0, 3.0, 4.0})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({4, 4}),
                        std::string("average"), std::vector<double>({5.0}))));

TEST(poolinglayer, new_pooling_layer_can_run_float_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F,
                            2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<float> true_output = {6.5F, 4.5F, 4.5F, 6.5F};
  EXPECT_EQ(*output.as<float>(), true_output);
}

TEST(poolinglayer, new_pooling_layer_can_run_int_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<int> true_output = {6, 4, 4, 6};
  EXPECT_EQ(*output.as<int>(), true_output);
}

TEST(poolinglayer, new_pooling_layer_can_run_1d_pooling_float) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<float> true_output = {8.0F, 5.0F, 2.5F};
  EXPECT_EQ(*output.as<float>(), true_output);
}
