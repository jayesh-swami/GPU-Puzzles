#include <hip/hip_runtime.h>
#include "helpers/hip_helpers.h"
#include <iostream>
#include <vector>

#define NUM_THREADS 256

using std::cout;
using std::vector;

__global__ void zip(int a[], int b[], int out[], int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size) out[idx] = a[idx] + b[idx];
}

// Useful when N is very large and numBlocks are too many
__global__ void zip_grid_stride(int a[], int b[], int out[], int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = idx; i < size; i += blockDim.x * gridDim.x)
		out[idx] = a[idx] + b[idx];
}
int main() {
	constexpr int N = 1024;

	vector<int> a_h(N, 10), b_h(N, 200);
	int *a_d, *b_d, *out_d;
	int out_h[N];

	HIP_CHECK(hipMalloc(&out_d, N * sizeof(int)));
	HIP_CHECK(hipMalloc(&a_d, N * sizeof(int)));
	HIP_CHECK(hipMalloc(&b_d, N * sizeof(int)));

	HIP_CHECK(H2D(a_d, a_h.data(), N * sizeof(int)));
	HIP_CHECK(H2D(b_d, b_h.data(), N * sizeof(int)));

	int numBlocks = (N + NUM_THREADS - 1) / NUM_THREADS;

	//zip<<<numBlocks, NUM_THREADS>>>(a_d, b_d, out_d, N);
	zip_grid_stride<<<4, NUM_THREADS>>>(a_d, b_d, out_d, N);

	HIP_CHECK(D2H(out_h, out_d, N * sizeof(int)));

	for(int i = 0; i < N; i++) {
		cout << out_h[i] << ' ';
	}

	cout << '\n';
	return 0;
}
