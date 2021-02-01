all:
	nvcc -g -lineinfo -I$(GFLAGS)/include fft.cu -lcufft -L$(GFLAGS)/lib -lgflags -o fft
clean:
	rm -f fft