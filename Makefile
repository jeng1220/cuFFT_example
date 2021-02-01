all:
	nvcc -g -lineinfo fft.cu -lcufft -o fft