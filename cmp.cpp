/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <gflags/gflags.h>

DEFINE_string(fn1, "", "file 1");
DEFINE_string(fn2, "", "file 2");

std::vector<float> read_bin(std::string fn) {
  std::ifstream is (fn.c_str(), std::ifstream::binary);
  assert(is && "invalid file name");

  is.seekg (0, is.end);
  int length = is.tellg();
  is.seekg (0, is.beg);

  std::vector<float> buf(length/sizeof(float));
  char* ptr = reinterpret_cast<char *>(buf.data());

  is.read(ptr, length);
  assert(is && "failed to read the file, file is corrupt");
  is.close();

  return buf;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  assert(FLAGS_fn1.length() > 0 && "invalid file name 1");
  assert(FLAGS_fn2.length() > 0 && "invalid file name 2");  

  std::cout << "compare " << FLAGS_fn1 << " and " << FLAGS_fn2 << std::endl;
  auto data1 = read_bin(FLAGS_fn1);
  auto data2 = read_bin(FLAGS_fn2);
  assert(data1.size() == data2.size() && "invalid comparison");

  
  for (size_t i = 0; i < data1.size(); ++i) {
    float x1 = data1[i];
    float x2 = data2[i];
    float delta = std::fabs(x1 - x2);
    if (delta > 1e-9) {
      fprintf(stderr, "data1[%zu] = %.9f (%x); data2[%zu] = %.9f (%x); delta = %.9f\n",
        i, x1, *(reinterpret_cast<int*>(&x1)), i, x2, *(reinterpret_cast<int*>(&x2)), delta);
    }
  }

  return 0;
}
