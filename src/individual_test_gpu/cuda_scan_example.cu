#include <iostream>

template <typename T>
__global__ void prescan(T *const g_odata, T *const g_idata, const int n,
			const T target_value) {
	// extern __shared__ T temp[]; // allocated on invocation
	// T *const mapped_idata = temp + n;
	extern __shared__ T mapped_idata[];
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	const auto thread_id = threadIdx.x;

	for (auto i = index; i < n; i += stride) {
		mapped_idata[i] = g_idata[i] == target_value;
	}
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= n; stride *= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if (index < 2 * n) {
			mapped_idata[index] += mapped_idata[index - stride];
		}
	}
	for (auto stride = n / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * n) {
			mapped_idata[index + stride] += mapped_idata[index];
		}
	}
	// Simple, but probably inefficient and redundant: write data to output.
	__syncthreads();
	for (auto i = index; i < n; i += stride) {
		g_odata[i] = mapped_idata[i];
	}

	// Old exclusive scan implementation, to be removed...
	// auto offset = 1;
	// temp[2 * thread_id] = mapped_idata[2 * thread_id]; // load input into shared memory
	// temp[2 * thread_id + 1] = mapped_idata[2 * thread_id + 1];
	// for (auto d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
	// 	__syncthreads();
	// 	if (thread_id < d) {
	// 		auto ai = offset * (2 * thread_id + 1) - 1;
	// 		auto bi = offset * (2 * thread_id + 2) - 1;
	// 		temp[bi] += temp[ai];
	// 	}
	// 	offset *= 2;
	// }
	// if (thread_id == 0) {
	// 	temp[n - 1] = 0; // clear the last element
	// }
	// for (auto d = 1; d < n; d *= 2) { // traverse down tree & build scan
	// 	offset >>= 1;
	// 	__syncthreads();
	// 	if (thread_id < d) {
	// 		auto ai = offset * (2 * thread_id + 1) - 1;
	// 		auto bi = offset * (2 * thread_id + 2) - 1;
	// 		T t = temp[ai];
	// 		temp[ai] = temp[bi];
	// 		temp[bi] += t;
	// 	}
	// }
	// __syncthreads();
	// g_odata[2 * thread_id] = temp[2 * thread_id]; // write results to device memory
	// g_odata[2 * thread_id + 1] = temp[2 * thread_id + 1];
}

int main() {
	return 0;
}
