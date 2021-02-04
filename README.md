# cuFFT example #
This is a simple example to demonstrate cuFFT usage. It will run 1D, 2D and 3D FFT complex-to-complex and save results with device name prefix as file name.


# build #
1. clone GFLAGS
```sh
$ git submodule init
$ git submodule update
```
2. build and install gflags
```sh
$ cd gflags
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=<path to installed gflags> ..
# for instance
$ cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install ..
$ make install
```
3. build the sample
```sh
$ GFLAGS=<path to installed gflags> CUDA=<path to CUDA> make
# for instance
$ GFLAGS=`pwd`/gflags/build/install CUDA=/usr/local/cuda make
```

# run #

fft, run 1D, 2D and 3D FFT on GPU
```sh
$ fft --help
  Flags from fft.cu:
    -batch_size (The batch size for 1D FFT) type: int32 default: 1
    -device_id (The device ID) type: int32 default: 0
    -nx (The transform size in the x dimension) type: int32 default: 64
    -ny (The transform size in the y dimension) type: int32 default: 64
    -nz (The transform size in the z dimension) type: int32 default: 64
```
cmp, compare results
```sh
$ cmp --help
  Flags from cmp.cpp:
    -fn1 (file 1) type: string default: ""
    -fn2 (file 2) type: string default: ""
```

