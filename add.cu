#include <iostream>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <cstdlib>


__global__ void vecAdd(float *a, float *b, float *C, int vectorLength) {
	int workindex = threadIdx.x + blockDim.x * blockIdx.x;
	if (workindex < vectorLength) {
		C[workindex] = a[workindex] + b[workindex];
	}

}

__global__ void mat3add(float *a, float *b, float *c, int nx, int ny, int nz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < nx && j < ny && k < nz) {
		int idx = i + j * nx + k * nx *ny;
		if (idx < nx * ny * nz) {
			c[idx] = a[idx] + b[idx];
		}
	}
}

void serialVecAdd(float *a, float *b, float *C, int vectorLength) {
	for (int i{0}; i < vectorLength; i ++) {
		C[i] = a[i] + b[i];
	}
}


int main() {
	
	int nx=100, ny = 100, nz = 1000;
	int vectorLength = nz * nx * ny;

	float *h_c = (float *)malloc(vectorLength * sizeof(float));
	float *h_a = (float *)malloc(vectorLength * sizeof(float));
	float *h_b = (float *)malloc(vectorLength * sizeof(float));
	float *d_a,*d_b,*d_c;

	cudaMalloc(&d_a, vectorLength * sizeof(float));
	cudaMalloc(&d_b, vectorLength * sizeof(float));
	cudaMalloc(&d_c, vectorLength * sizeof(float));

	for (int i{0}; i < vectorLength; i ++ ) {
		h_a[i] = i;
		h_b[i] = 2 * i;
	}

	cudaMemcpy(d_a, h_a, sizeof(float) * vectorLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float) * vectorLength, cudaMemcpyHostToDevice);
	
	dim3 blocksize(8, 8, 16);
	dim3 numBlocks(cuda::ceil_div(nx, blocksize.x), cuda::ceil_div(ny, blocksize.y), cuda::ceil_div(nz, blocksize.z));
	mat3add<<<numBlocks, blocksize>>>(d_a,d_b, d_c, nx, ny, nz);

	cudaDeviceSynchronize();

	cudaMemcpy(h_c, d_c, sizeof(float) * vectorLength, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);
}
