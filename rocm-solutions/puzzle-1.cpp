#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "helpers/hip_helpers.h"

using std::vector;
using std::cout;

__global__ void addNKernel(int vec[], int n, int size) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < size) vec[i] = vec[i] + n;
}

int main() {
	constexpr int N = 64;

	int v_h[N];
	int *v_d;

	for(int i = 0; i < N; i++) v_h[i] = i;

	HIP_CHECK(hipMalloc(&v_d, N * sizeof(int)));

	HIP_CHECK(hipMemcpy(v_d, v_h, N * sizeof(int), hipMemcpyHostToDevice));

	addNKernel<<< 1, N >>>(v_d, 10, N);

	hipError_t err = hipMemcpy(v_h, v_d, N * sizeof(int), hipMemcpyDeviceToHost);
	
	if(err != hipSuccess) {
		cout << "ERROR: " << hipGetErrorString(err) << '\n';
	} else {
		for(int i = 0; i < N; i++) {
			cout << v_h[i] << '\n';
		}
	}

	hipFree(v_d);

	return 0;
}
