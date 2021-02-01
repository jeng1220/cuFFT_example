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

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void check(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << ", line: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void check(cufftResult err, int line) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "Error: " << err << ", line: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::string remove_space(char *str) {
    char *ix = str;
    int n = 0;
    while((ix = strchr(ix, ' ')) != NULL) {
        *ix++ = '_';
        n++;
    }
    std::string name(str);
    return name + std::string("_");
}

float my_rand() {
#if 1
  static float tmp[] = {
    0.088661, 0.719976, 0.355909, 0.268335, 0.216837,
    0.111352, 0.722319, 0.834714, 0.855696, 0.464120, 
    0.125076, 0.427663, 0.361393, 0.494953, 0.145556};
  static unsigned long count = 0;
  size_t size = sizeof(tmp)/sizeof(float);
  count++;
  return tmp[count % size];
#else
  return static_cast<float>(rand()) / RAND_MAX;
#endif
}

#define CHECK(x) check((x), __LINE__)

template <typename T>
void write(const std::string& fn, T vec) {
  int dev_id;
  CHECK(cudaGetDevice(&dev_id));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev_id));
  auto dev_name = remove_space(prop.name);

  std::fstream fs;
  fs.open(dev_name + fn + std::string(".bin"), std::fstream::out);
  auto* h_ptr = reinterpret_cast<char *>(thrust::raw_pointer_cast(vec.data()));
  size_t mem_size = vec.size() * sizeof(cufftComplex);
  fs.write(h_ptr, mem_size);
  fs.close();
}

void c2c(int nx, int ny, int nz, int direction, const std::string& fn) {
  size_t elements = nx * ny * nz;
  thrust::host_vector< cufftComplex > h_in_signal(elements);
  for (size_t i = 0; i < elements; ++i) {
    h_in_signal[i] = cufftComplex{my_rand(), my_rand()};
  }

  thrust::host_vector< cufftComplex > h_out_signal(elements);
  thrust::fill(h_out_signal.begin(), h_out_signal.end(), cufftComplex{0.f, 0.f});

  thrust::device_vector< cufftComplex > d_in_signal(elements);
  d_in_signal = h_in_signal;
  thrust::device_vector< cufftComplex > d_out_signal(elements);
  thrust::fill(d_out_signal.begin(), d_out_signal.end(), cufftComplex{0.f, 0.f});

  cufftHandle plan;
  if (ny == nz && ny == 1) {
    CHECK(cufftPlan1d(&plan, nx, CUFFT_C2C, 1));
  }
  else if (ny != 1 && nz == 1) {
    CHECK(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
  }
  else if (ny != 1 && nz != 1) {
    CHECK(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));
  }

  auto* d_in_ptr = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(d_in_signal.data()));
  auto* d_out_ptr = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(d_out_signal.data()));

  CHECK(cufftExecC2C(plan, d_in_ptr, d_out_ptr, direction));

  h_out_signal = d_out_signal;

  write(fn, h_out_signal);
  CHECK(cufftDestroy(plan));
}

void c2c1d_fwd(int nx) {
  c2c(nx, 1, 1, CUFFT_FORWARD, std::string(__func__));
}

void c2c2d_fwd(int nx, int ny) {
  c2c(nx, ny, 1, CUFFT_FORWARD, std::string(__func__));
}

void c2c3d_fwd(int nx, int ny, int nz) {
  c2c(nx, ny, nz, CUFFT_FORWARD, std::string(__func__));
}

void c2c1d_inv(int nx) {
  c2c(nx, 1, 1, CUFFT_INVERSE, std::string(__func__));
}

void c2c2d_inv(int nx, int ny) {
  c2c(nx, ny, 1, CUFFT_INVERSE, std::string(__func__));
}

void c2c3d_inv(int nx, int ny, int nz) {
  c2c(nx, ny, nz, CUFFT_INVERSE, std::string(__func__));
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  int device = (argc > 1) ? atoi(argv[1]):0;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU = " << prop.name << std::endl;
  CHECK(cudaSetDevice(device));

  c2c1d_fwd(100);
  c2c2d_fwd(100, 100);
  c2c3d_fwd(100, 100, 100);

  c2c1d_inv(100);
  c2c2d_inv(100, 100);
  c2c3d_inv(100, 100, 100);

  return 0;
}
