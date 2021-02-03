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
$ make install
```
3. build the sample (fft.cu)
```sh
$ GFLAGS=<path to gflags> make
```

# run #
```sh
  Flags from fft.cu:
    -device_id (The device ID) type: int32 default: 0
    -nx (The transform size in the x dimension) type: int32 default: 64
    -ny (The transform size in the y dimension) type: int32 default: 64
    -nz (The transform size in the z dimension) type: int32 default: 64
```
