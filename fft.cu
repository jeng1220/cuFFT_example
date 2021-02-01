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

int remove_space(char *str) {
    char *ix = str;
    int n = 0;
    while((ix = strchr(ix, ' ')) != NULL) {
        *ix++ = '_';
        n++;
    }
    return n;
}

float my_rand() {
#if 0
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
#define SIGNAL_SIZE 16384

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{
  int device = (argc > 1) ? atoi(argv[1]):0;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device));
  remove_space(prop.name);
  printf("GPU=%s\n", prop.name);
  CHECK(cudaSetDevice(device));

  int width = 8196;
  int height = 8196;
  size_t elements = width * height;

  // Allocate host memory for the signal
  cufftComplex *h_signal = new cufftComplex[elements];

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = my_rand();
    h_signal[i].y = my_rand();
  }

  // Allocate device memory for signal
  cufftComplex *d_signal;
  size_t mem_size = sizeof(cufftComplex) * elements;
  CHECK(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
  // Copy host memory to device
  CHECK(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

  // CUFFT plan simple API
  cufftHandle plan;
  CHECK(cufftPlan2d(&plan, width, height, CUFFT_C2C));

  // Transform signal and kernel
  CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
    reinterpret_cast<cufftComplex *>(d_signal),
    CUFFT_FORWARD));

  // Check if kernel execution generated and error
  CHECK(cudaStreamSynchronize(0));
  CHECK(cudaGetLastError());

  // Transform signal back
  CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
    reinterpret_cast<cufftComplex *>(d_signal),
    CUFFT_INVERSE));

  // Copy device memory to host
  CHECK(cudaMemcpy(h_signal, d_signal, mem_size,
    cudaMemcpyDeviceToHost));

  printf("h_signal[0]=%.9f\n", h_signal[0].x);
  printf("h_signal[0]=%.9f\n", h_signal[0].y);
  printf("h_signal[100]=%.9f\n", h_signal[100].x);
  printf("h_signal[100]=%.9f\n", h_signal[100].y);
  std::fstream fs;
  fs.open(prop.name, std::fstream::out);
  fs.write(reinterpret_cast<char *>(h_signal), mem_size);
  fs.close();

  // Destroy CUFFT context
  CHECK(cufftDestroy(plan));

  // cleanup memory
  delete [] h_signal;
  CHECK(cudaFree(d_signal));

  return 0;
}
