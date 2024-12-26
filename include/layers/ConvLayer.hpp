#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class ConvolutionalLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  Tensor kernel_;
  Tensor bias_;

 public:
  ConvolutionalLayer() = default;
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations,
                     const Tensor& kernel, const Tensor& bias = Tensor()) {
    stride_ = step;
    pads_ = pads;
    dilations_ = dilations;
    kernel_ = kernel;
    bias_ = bias;
  }

  void run(const Tensor& input, Tensor& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return kernel_; }
#endif
};

template <typename ValueType>
class ConvImpl : public LayerImpl<ValueType> {
 private:
  int input_width_;
  int input_height_;
  int input_flow_;
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  size_t input_size_;
  std::vector<ValueType> bias_;

 public:
  ConvImpl() = delete;
  ConvImpl(size_t stride, size_t pads, size_t dilations, int input_width,
           int input_height, int input_flow, size_t input_size,
           const std::vector<ValueType>& bias)
      : input_width_(input_width),
        input_height_(input_height),
        input_flow_(input_flow),
        stride_(stride),
        pads_(pads),
        dilations_(dilations),
        input_size_(input_size),
        bias_(bias) {}

  ConvImpl(const ConvImpl& c) = default;

  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override {
    return input;
  }

  std::vector<ValueType> run(std::vector<ValueType> startmatrix, int new_rows,
                             int new_cols, std::vector<ValueType> startkernel,
                             size_t start_kernel_size, size_t kernel_size,
                             int center_distance) const {
    std::vector<ValueType> matrix(new_rows * new_cols * input_flow_, 0);
    for (int i = 0; i < input_height_; ++i) {
      for (int j = 0; j < input_width_; ++j) {
        for (int f = 0; f < input_flow_; ++f) {
          matrix[((i + pads_) * new_cols + j + pads_) * input_flow_ + f] =
              startmatrix[(i * input_width_ + j) * input_flow_ + f];
        }
      }
    }

    std::vector<ValueType> kernel(kernel_size * kernel_size, 0);
    for (int i = 0; i < static_cast<int>(start_kernel_size); ++i) {
      for (int j = 0; j < static_cast<int>(start_kernel_size); ++j) {
        kernel[(dilations_ + i) * static_cast<int>(kernel_size) + j +
               (j + 1) * dilations_] =
            startkernel[i * static_cast<int>(start_kernel_size) + j];
      }
    }

    std::vector<ValueType> outputvec;
    for (int i = input_width_ + center_distance;
         i < static_cast<int>(input_size_); i += static_cast<int>(stride_)) {
      for (int x = 0; x < input_flow_; ++x) {
        ValueType color = 0;
        for (int coloms = -input_width_; coloms < input_width_ + 1;
             coloms += input_width_) {
          for (int str = -1; str < 2; ++str) {
            if (input_width_ == 0) {
              throw std::out_of_range("Input = 0");
            }
            auto kercol = static_cast<size_t>(coloms / input_width_ + 1);
            color +=
                matrix[(i + coloms + str) * input_flow_ + x] *
                kernel[kercol * kernel_size + static_cast<size_t>(str + 1)];
          }
        }
        if (!bias_.empty() && static_cast<size_t>(x) < bias_.size()) {
          color += bias_[x];
        }
        outputvec.push_back(color);
      }
      if ((i + center_distance + 1) % input_width_ == 0) {
        if (i + input_width_ + center_distance * 2 ==
            static_cast<int>(input_size_)) {
          i += input_width_ + center_distance * 2 + 1;
        } else {
          i += input_width_ * (static_cast<int>(stride_) - 1) +
               (3 - static_cast<int>(stride_));
        }
      }
    }
    return outputvec;
  }
};

// NCHW -> NCHW only
template <typename ValueType>
void Conv4D(const Tensor& input, const Tensor& kernel_, const Tensor& bias_,
              Tensor& output, size_t stride_,
              size_t pads_, size_t dilations_) {
  size_t batch_size = input.get_shape()[0];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];
  size_t in_channels = input.get_shape()[1];

  size_t kernel_height = kernel_.get_shape()[0];
  size_t kernel_width = kernel_.get_shape()[1];
  size_t kernel_in_channels = kernel_.get_shape()[2];
  size_t kernel_out_channels = kernel_.get_shape()[3];

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> input_tensor(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          in_height, std::vector<std::vector<ValueType>>(
                         in_width, std::vector<ValueType>(in_channels, 1))));
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t c = 0; c < in_channels; c++) {
      for (size_t h = 0; h < in_height; h++) {
        for (size_t w = 0; w < in_width; w++) {
          input_tensor[n][h][w][c] = input.get<ValueType>({n, c, h, w});
        }
      }
    }
  }
  // adapt input

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> kernel(
      kernel_height,
      std::vector<std::vector<std::vector<ValueType>>>(
          kernel_width,
          std::vector<std::vector<ValueType>>(
                            kernel_in_channels,
                            std::vector<ValueType>(kernel_out_channels, 1))));
  for (size_t h = 0; h < kernel_height; h++) {
    for (size_t w = 0; w < kernel_width; w++) {
      for (size_t n = 0; n < kernel_in_channels; n++) {
        for (size_t c = 0; c < kernel_out_channels; c++) {
          kernel[h][w][n][c] = kernel_.get<ValueType>({h, w, n, c});
        }
      }
    }
  }
  // adapt kernel

  // pads_ = (kernel_height * (-1 + 2 * dilations_) - 1) / 2;
  // ???

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input =
      input_tensor;
  if (pads_ > 0) {
    padded_input =
        std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
        batch_size,
            std::vector<std::vector<std::vector<ValueType>>>(
            in_height + 2 * pads_,
                std::vector<std::vector<ValueType>>(
                                in_width + 2 * pads_,
                                std::vector<ValueType>(in_channels, 0))));

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < in_height; ++h) {
        for (size_t w = 0; w < in_width; ++w) {
          for (size_t c = 0; c < in_channels; ++c) {
            padded_input[b][h + pads_][w + pads_][c] = input_tensor[b][h][w][c];
          }
        }
      }
    }
  }
  // | | | | |
  // | data  |
  // | | | | |

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel =
      kernel;
  if (dilations_ > 1) {
    dil_kernel = std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
        kernel_height * (-1 + 2 * dilations_),
        std::vector<std::vector<std::vector<ValueType>>>(
            kernel_width * (-1 + 2 * dilations_),
            std::vector<std::vector<ValueType>>(
                kernel_in_channels,
                std::vector<ValueType>(kernel_out_channels, 0))));

    for (size_t b = 0; b < kernel_out_channels; ++b) {
      for (size_t h = 0; h < kernel_height; ++h) {
        for (size_t w = 0; w < kernel_width; ++w) {
          for (size_t c = 0; c < kernel_in_channels; ++c) {
            dil_kernel[(h * (-1 + 2 * dilations_))][(w * (-1 + 2 * dilations_))]
                      [c][b] = kernel[h][w][c][b];
          }
        }
      }
    }
  }

  size_t crat = 0;
  if ((in_height + 2 * pads_ - ((kernel_height * (-1 + 2 * dilations_)) - 1)) %
          stride_ !=
      0)
    crat = 1;

  size_t out_height =
      (in_height + 2 * pads_ - ((kernel_height * (-1 + 2 * dilations_)) - 1)) /
          stride_ +
      crat;

  crat = 0;

  if ((in_width + 2 * pads_ - ((kernel_width * (-1 + 2 * dilations_)) - 1)) %
          stride_ !=
      0)
    crat = 1;

  size_t out_width =
      (in_width + 2 * pads_ - ((kernel_width * (-1 + 2 * dilations_)) - 1)) /
          stride_ +
      crat;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                      kernel_out_channels,
                      std::vector<std::vector<ValueType>>(
                          out_height, std::vector<ValueType>(out_width, 0))));
  size_t one_size = (kernel_height * (-1 + 2 * dilations_) - 1) / 2;

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < kernel_out_channels; ++c) {
      for (size_t i = 0; i < out_height; i += stride_) {
        for (size_t j = 0; j < out_width; j += stride_) {
          ValueType value = 0;
          for (size_t ic = 0; ic < in_channels; ++ic) {
            for (int h = (-1 * static_cast<int>(one_size));
                 h <= static_cast<int>(one_size); ++h) {
              for (int w = (-1 * static_cast<int>(one_size));
                   w <= static_cast<int>(one_size); ++w) {
                value +=
                    padded_input[b][i + one_size + h][j + one_size + w][ic] *
                    dil_kernel[one_size + h][one_size + w][ic][c];
              }
            }
          }
          output_tensor[b][c][i][j] = value + (*bias_.as<ValueType>())[c];
        }
      }
    }
  }

  Shape sh({batch_size, kernel_out_channels, out_height, out_width});
  std::vector<ValueType> one_d_vector(batch_size * out_height * out_width *
                                  kernel_out_channels);
  size_t index_1d = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t l = 0; l < kernel_out_channels; ++l) {
      for (size_t j = 0; j < out_height; ++j) {
        for (size_t k = 0; k < out_width; ++k) {
          one_d_vector[index_1d++] = output_tensor[i][l][j][k];
        }
      }
    }
  }
  output = make_tensor<ValueType>(one_d_vector, sh);
}

}  // namespace itlab_2023