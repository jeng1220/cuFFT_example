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
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


DEFINE_int32(device_id, 0, "The device ID");
DEFINE_int32(batch_size, 1, "The batch size for 1D FFT");
DEFINE_int32(nx, 64, "The transform size in the x dimension");
DEFINE_int32(ny, 64, "The transform size in the y dimension");
DEFINE_int32(nz, 64, "The transform size in the z dimension");

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

#define CHECK(x) check((x), __LINE__)

std::string remove_space(char *str) {
    char *ix = str;
    int n = 0;
    while((ix = strchr(ix, ' ')) != NULL) {
        *ix++ = '_';
        n++;
    }
    std::string name(str);
    return name;
}

float my_rand() {
#if 1
  static float tmp[] = {
    0.184393838, 0.267245561, 0.911512017, 0.408272356, 0.146058381,
    0.695602775, 0.391118735, 0.848621726, 0.617213607, 0.122606553,
    0.190755174, 0.822026193, 0.156823128, 0.489058524};
  static unsigned long count = 0;
  size_t size = sizeof(tmp)/sizeof(float);
  count++;
  return tmp[count % size];
#else
  return static_cast<float>(rand()) / RAND_MAX;
#endif
}

template <typename T>
void write(const std::string& fn, T vec) {
  int dev_id;
  CHECK(cudaGetDevice(&dev_id));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev_id));
  auto dev_name = remove_space(prop.name);

  std::string cufft_ver("_cuFFT_");
  cufft_ver += std::to_string(CUFFT_VERSION);
  cufft_ver += std::to_string(CUFFT_VER_BUILD);
  cufft_ver += std::string("_");

  std::string nvcc_ver("_NVCC_");
  nvcc_ver += std::to_string(__CUDACC_VER_MAJOR__);
  nvcc_ver += std::to_string(__CUDACC_VER_MINOR__);
  nvcc_ver += std::to_string(__CUDACC_VER_BUILD__);

  std::fstream fs;
  fs.open(dev_name + nvcc_ver + cufft_ver + fn + std::string(".bin"), std::fstream::out);
  auto* h_ptr = reinterpret_cast<char *>(thrust::raw_pointer_cast(vec.data()));
  size_t mem_size = vec.size() * sizeof(cufftComplex);
  fs.write(h_ptr, mem_size);
  fs.close();
}

void c2c(int nx, int ny, int nz, int direction, const std::string& fn) {
  assert(nx > 0 && ny > 0 && nz > 0 && "nx, ny, nz need to be greater than 0");
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
    CHECK(cufftPlan1d(&plan, nx, CUFFT_C2C, FLAGS_batch_size));
  }
  else if (ny != 1 && nz == 1) {
    CHECK(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
  }
  else if (ny != 1 && nz != 1) {
    CHECK(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));
  }
  else {
    assert(!"should not be here");
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
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(cudaSetDevice(FLAGS_device_id));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, FLAGS_device_id));
  std::cout << "GPU = " << prop.name << std::endl;

  c2c1d_fwd(FLAGS_nx);
  c2c2d_fwd(FLAGS_nx, FLAGS_ny);
  c2c3d_fwd(FLAGS_nx, FLAGS_ny, FLAGS_nz);

  c2c1d_inv(FLAGS_nx);
  c2c2d_inv(FLAGS_nx, FLAGS_ny);
  c2c3d_inv(FLAGS_nx, FLAGS_ny, FLAGS_nz);

  return 0;
}
