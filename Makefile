all:
	nvcc -g -lineinfo -I./gflags/build/out/include fft.cu -lcufft -L./gflags/build/out/lib -lgflags -o fft