#include <random>
#include <thread>
#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/FCLayer.hpp"
#include "perf/benchmarking.hpp"

TEST(basic, basic_test) {
  // Arrange
  int a = 2;
  int b = 3;

  // Act
  int c = a + b;

  // Assert
  ASSERT_EQ(5, c);
}
TEST(graph, check_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_connection1) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_connection_when_not_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_connection_when_not_connection2) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_connection_when_not_connection3) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}

// ==========================
// Fully Connected layer

TEST(fclayer, calculates_correctly1) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  std::vector<double> input = {1, 2};
  Shape2D<double> output = layer.run(input);
  EXPECT_NEAR(output.get(0), 5.5, 1e-5);
  EXPECT_NEAR(output.get(1), 4.4, 1e-5);
  EXPECT_NEAR(output.get(2), 12.0, 1e-5);
}

TEST(fclayer, calculates_correctly2) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  std::vector<double> input = {0.5, 0.0};
  Shape2D<double> output = layer.run(input);
  EXPECT_NEAR(output.get(0), 1.5, 1e-5);
  EXPECT_NEAR(output.get(1), 0.55, 1e-5);
  EXPECT_NEAR(output.get(2), 1.0, 1e-5);
}

TEST(fclayer, throws_when_greater_input_size) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  std::vector<double> input = {2.0, 1.0, 0.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_less_input_size) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  std::vector<double> input = {2.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}

TEST(fclayer, throws_when_empty_input) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  std::vector<double> input;
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_empty_bias) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias;
  ASSERT_ANY_THROW(FCLayer<double>(weights, bias));
}

TEST(fclayer, set_get_weight_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  for (size_t i = 0; i < weights.get_height(); i++) {
    for (size_t j = 0; j < weights.get_width(); j++) {
      EXPECT_NEAR(layer.get_weight(i, j), weights.get(i, j), 1e-5);
    }
  }
  for (size_t i = 0; i < weights.get_height(); i++) {
    for (size_t j = 0; j < weights.get_width(); j++) {
      layer.set_weight(i, j, i + j);
      EXPECT_NEAR(layer.get_weight(i, j), i + j, 1e-5);
    }
  }
}
TEST(fclayer, set_get_bias_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  for (size_t i = 0; i < bias.size(); i++) {
    EXPECT_NEAR(layer.get_bias(i), bias[i], 1e-5);
  }
  for (size_t i = 0; i < bias.size(); i++) {
    layer.set_bias(i, i);
    EXPECT_NEAR(layer.get_bias(i), i, 1e-5);
  }
}

TEST(fclayer, set_get_weight_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape2D<double> weights(3, 3, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.get_weight(4, 0));
  ASSERT_ANY_THROW(layer.get_weight(0, 4));
  ASSERT_ANY_THROW(layer.set_weight(4, 0, 1.3));
  ASSERT_ANY_THROW(layer.set_weight(0, 4, 1.3));
}
TEST(fclayer, set_get_bias_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape2D<double> weights(3, 3, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.get_bias(4));
  ASSERT_ANY_THROW(layer.set_bias(4, 1.3));
}

TEST(fclayer, get_dims_test1) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape2D<double> weights(3, 2, a1);
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  EXPECT_EQ(layer.get_dims().first, 3);
  EXPECT_EQ(layer.get_dims().second, 2);
}
TEST(fclayer, get_dims_test2) {
  FCLayer<double> layer;
  EXPECT_EQ(layer.get_dims().first, 0);
  EXPECT_EQ(layer.get_dims().second, 0);
}

// ==========================

// ==========================
// Timer tests

void waitfor_function(const size_t ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// chrono
TEST(timer, is_elapsed_time_returns_positive_value) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}
TEST(timer, is_elapsed_time_avg_returns_positive_value) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}

// omp
TEST(timer, is_elapsed_time_omp_returns_positive_value) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time_omp(waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}
TEST(timer, is_elapsed_time_omp_avg_returns_positive_value) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_omp_avg(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}

// >= -100 ms, <= +1000 ms
// chrono
TEST(timer, is_elapsed_time_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 1250);
}
TEST(timer, is_elapsed_time_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 1250);
}

// omp
TEST(timer, is_elapsed_time_omp_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time_omp(waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 1.25);
}
TEST(timer, is_elapsed_time_omp_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_omp_avg(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 1.25);
}

// ==========================

// ==========================
// Accuracy tests

TEST(accuracy, max_accuracy_test) {
  double a[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double b[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  auto acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 0.0, 1e-5);
}

TEST(accuracy, bad_accuracy_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  auto acc = accuracy<double>(a, b, 2);
  EXPECT_NEAR(acc, 99.3, 1e-5);
}

TEST(accuracy, bad_accuracy_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  auto acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 122.27, 1e-5);
}

TEST(accuracy, bad_accuracy_test_L) {
  size_t n = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < n; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < n; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < n; i++) {
    actual_acc += std::abs(a[i] - b[i]);
  }
  auto acc = accuracy<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  auto acc = accuracy_norm<double>(a, b, 2);
  EXPECT_NEAR(acc, 98.10734, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  auto acc = accuracy_norm<double>(a, b, 10);
  EXPECT_NEAR(acc, 52.72274, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_L) {
  size_t n = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < n; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < n; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < n; i++) {
    actual_acc += (a[i] - b[i]) * (a[i] - b[i]);
  }
  actual_acc = std::sqrt(actual_acc);
  auto acc = accuracy_norm<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, accuracy_throws_when_bad_pointer) {
  double *a = nullptr;
  auto *b = new double[5];
  EXPECT_ANY_THROW(accuracy<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy<double>(b, a, 5));
  delete[] b;
}

TEST(accuracy, accuracy_norm_throws_when_bad_pointer) {
  double *a = nullptr;
  auto *b = new double[5];
  EXPECT_ANY_THROW(accuracy_norm<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy_norm<double>(b, a, 5));
  delete[] b;
}

// ==========================

// ==========================
// Throughput tests

template <typename T>
std::vector<T> matrix_sum(const std::vector<T> &first,
                          const std::vector<T> &second) {
  std::vector<T> res(first);
  for (size_t i = 0; i < first.size(); i++) {
    res[i] += second[i];
  }
  return res;
}

template <typename T>
std::vector<T> matrix_mul(const size_t n, const std::vector<T> &first,
                          const std::vector<T> &second) {
  std::vector<T> mul(n * n, T(0));
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < n; k++) {
      for (size_t j = 0; j < n; j++) {
        mul[n * i + j] += first[n * i + k] * second[n * k + j];
      }
    }
  }
  return mul;
}

TEST(throughput, matrix_operations_throughput_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = i + j;
      b[ptr] = i - j;
      ptr++;
    }
  }
  double tp;
  tp = throughput<double, std::ratio<1, 1> >(matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput<double, std::ratio<1, 1> >(matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_avg_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = i + j;
      b[ptr] = i - j;
      ptr++;
    }
  }
  double tp;
  tp = throughput_avg<double, std::ratio<1, 1> >(10, matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_avg<double, std::ratio<1, 1> >(10, matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_omp_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = i + j;
      b[ptr] = i - j;
      ptr++;
    }
  }
  double tp;
  tp = throughput_omp(matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_omp(matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_omp_avg_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = i + j;
      b[ptr] = i - j;
      ptr++;
    }
  }
  double tp;
  tp = throughput_omp_avg(10, matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_omp_avg(10, matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}

// ==========================
