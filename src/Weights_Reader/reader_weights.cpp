﻿#include "Weights_Reader/reader_weights.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

// Функция для чтения JSON файла
json read_json(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + filename);
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  if (size == 0) {
    ifs.close();
    throw std::runtime_error("JSON file is empty: " + filename);
  }
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  ifs.read(buffer.data(), size);
  ifs.close();

  json model_data;
  try {
    model_data = json::parse(buffer.begin(), buffer.end());
  } catch (const json::parse_error& e) {
    throw std::runtime_error("JSON parse error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Standard exception: " + std::string(e.what()));
  } catch (...) {
    throw std::runtime_error("An unknown error occurred while parsing JSON.");
  }

  return model_data;
}

// Функция для извлечения значений из JSON
void extract_values_from_json(const json& j, std::vector<float>& values) {
  if (j.is_array()) {
    for (const auto& item : j) {
      extract_values_from_json(item, values);
    }
  } else if (j.is_number()) {  // Изменено на j.is_number()
    values.push_back(j.get<float>());
  } else if (!j.is_null()) {
    throw std::runtime_error("Unexpected type in JSON structure: " +
                             std::string(j.type_name()));
  }
}

void extract_values_without_bias(const json& j, std::vector<float>& values) {
  std::vector<float> temp_values;

  // Извлечение всех значений из JSON
  extract_values_from_json(j, temp_values);

  // Определяем размер bias, если он присутствует
  size_t bias_size = 0;
  if (j.is_array() && !j.empty() && j.back().is_array()) {
    bias_size =
        j.back().size();  // Размер bias равен размеру последнего массива
  }

  std::cout << "Temp values size: " << temp_values.size() << std::endl;
  std::cout << "Bias size: " << bias_size << std::endl;

  // Заполняем values значениями без bias
  if (temp_values.size() >= bias_size) {
    values.assign(temp_values.begin(), temp_values.end() - bias_size);
  } else {
    throw std::runtime_error("Extracted values are smaller than bias size.");
  }
  std::cout << "Values size after extraction: " << values.size() << std::endl;
}

// Функция для определения формы из JSON
void parse_json_shape(const json& j, std::vector<size_t>& shape, size_t dim) {
  // Если dim == 0, просто проверяем первый уровень
  if (dim == 0) {
    if (j.is_array()) {
      if (j.empty()) {
        // Пустой массив
        shape.push_back(0);
        return;
      }
      // Рекурсивно вызываем функцию на первом элементе массива
      parse_json_shape(j.front(), shape, dim + 1);
    } else {
      // Если на первом уровне нет массива, задаем пустую форму
      shape.push_back(0);
    }
  } else {
    if (j.is_array()) {
      if (shape.size() <= dim - 1) {
        shape.push_back(j.size());
      } else if (shape[dim - 1] != j.size()) {
        throw std::runtime_error("Inconsistent array size at dimension " +
                                 std::to_string(dim - 1));
      }
      // Рекурсивно вызываем функцию на первом элементе массива, если массив не
      // пуст
      if (!j.empty()) {
        parse_json_shape(j.front(), shape, dim + 1);
      }
    } else if (!j.is_number() && !j.is_null()) {
      throw std::runtime_error("Unexpected type in JSON structure: " +
                               std::string(j.type_name()));
    }
  }
}

void extract_bias_from_json(const json& j, std::vector<float>& bias) {
  if (j.is_array()) {
    if (!j.empty() && j.back().is_array()) {
      for (const auto& item : j.back()) {
        if (item.is_number()) {
          bias.push_back(item.get<float>());
        } else {
          throw std::runtime_error("Unexpected type in bias array: " +
                                   std::string(item.type_name()));
        }
      }
    }
  } else {
    throw std::runtime_error("Input JSON structure should be an array.");
  }
}

// Функция для создания тензора из JSON
Tensor create_tensor_from_json(const json& j, Type type) {
  if (type == Type::kFloat) {
    std::vector<float> vals;
    std::vector<size_t> shape;
    std::vector<float> bias;

    // Извлечение значений из JSON
    extract_values_without_bias(j, vals);
    std::cout << "Extracted values size: " << vals.size() << std::endl;

    // Определение формы тензора
    parse_json_shape(j, shape);
    std::cout << "Parsed shape: ";
    size_t expected_size = 1;
    for (const auto& dim : shape) {
      std::cout << dim << " ";
      expected_size *= dim;
    }
    std::cout << std::endl;

    // Обработка пустых слоев
    if (expected_size == 1 && shape.empty()) {
      expected_size = 0;
    }
    std::cout << "Expected size: " << expected_size << std::endl;
    try {
      extract_bias_from_json(j, bias);
      std::cout << "Extracted bias size: " << bias.size() << std::endl;
    } catch (const std::exception& e) {
      std::cout << "No bias found or error extracting bias: " << e.what()
                << std::endl;
    }
    Shape sh(shape);
    return make_tensor<float>(vals, sh, bias);
  }
  throw std::invalid_argument("Unsupported type or invalid JSON format");
}
