#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;
TEST(Read_img, can_read_image) {
  ASSERT_NO_THROW(Mat image = imread("image.jpg"););
}
TEST(Read_img, can_save_image) {
  Mat image = imread("image.jpg");
  String outputFileName = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(outputFileName, image););
}
