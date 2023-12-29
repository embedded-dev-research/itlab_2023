﻿#include "reader_img.hpp"
void read(std::string path) {
  Mat image = imread(path);
  if (image.empty()) {
    throw "Could not open or find the image";
    cin.get();
  }
  String windowName = "Image";
  namedWindow(windowName);
  imshow(windowName, image);
  waitKey(0);
  destroyWindow(windowName);
}