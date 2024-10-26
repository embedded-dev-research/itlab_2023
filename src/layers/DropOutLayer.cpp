#include "layers/DropOutLayer.hpp"

#include <algorithm>
#include <functional>
#include <random>

namespace itlab_2023 {

void DropOutLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      std::vector<int> vec = *input.as<int>();
      const double lower_bound = 0;
      const double upper_bound = 100;
      std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      for (size_t i = 0; i < vec.size(); i++) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100)
          vec[i] = 0;
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    case Type::kFloat: {
      std::vector<float> vec = *input.as<float>();
      const double lower_bound = 0;
      const double upper_bound = 100;
      std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      for (size_t i = 0; i < vec.size(); i++) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100)
          vec[i] = 0;
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace itlab_2023