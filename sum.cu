#include <iostream>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <cstdlib>


__global__ void Sum(float *a,  float *C, int vectorLength) {
	extern __shared__ float reduc[vectorLength / 2];
	int currentSize = vectorLength;
	for (;currentSize > 1;currentSize /= 2) {
		int loc =  blockDim.x * blockIdx.x + threadIdx.x * 2;
		if (loc < currentSize) {
			reduc[loc] = a[loc] + a[loc+1];
		}
		__syncthreads();
	}
}

int main() {
	
	int vectorLength = 512;

	float *a = nullptr;
	float *C = nullptr;

	float *comparison = (float *)malloc(vectorLength * sizeof(float));

	cudaMallocManaged(&a, vectorLength * sizeof(float));
	cudaMallocManaged(&C, sizeof(float));
	
	for (int i{0}; i < vectorLength; i ++ ) {
		a[i] = i;
	}
	
	int threads = 256;
	// int blocks = cuda::ceil_div(vectorLength, threads);
	//vecAdd<<<blocks, threads>>>(a,b, C, vectorLength);

	cudaDeviceSynchronize();
	
	cudaFree(a);
	cudaFree(C);
	free(comparison);
}
