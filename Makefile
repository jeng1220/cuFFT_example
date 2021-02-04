all:fft cmp

fft:fft.cu
	$(CUDA)/bin/nvcc -g -lineinfo -I$(GFLAGS)/include -I$(CUDA)/include fft.cu -L$(CUDA)/lib64 -lcufft -L$(GFLAGS)/lib -lgflags -o fft

cmp:cmp.cpp
	g++ -g -I$(GFLAGS)/include cmp.cpp -L$(GFLAGS)/lib -lgflags -lpthread -o cmp

clean:
	rm -f fft

