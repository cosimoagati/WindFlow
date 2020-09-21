#include <cstdlib>
#include <iostream>
#include <vector>
#include "../../wf/gpu_utils.hpp"

using namespace std;
using wf::GPUBuffer;

static constexpr auto threads_per_block = 512;

__global__ void map_to_target(int *const output, int *const input, const int n, const int target_value) {
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	if (absolute_id < n)
		output[absolute_id] = input[absolute_id] == target_value;
}

__global__ void prescan(int *const output, int *const input, int *const partial_sums, const int n) {
	extern __shared__ int temp[];
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	const auto block_thread_id = threadIdx.x;

	if (absolute_id < n)
		temp[block_thread_id] = input[absolute_id];
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= n; stride *= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index < 2 * n)
			temp[index] += temp[index - stride];
	}
	for (auto stride = n / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index + stride < 2 * n)
			temp[index + stride] += temp[index];
	}
	__syncthreads();
	if (block_thread_id == 0 && blockIdx.x > 0)
		partial_sums[blockIdx.x - 1] = temp[n - 1];
	if (absolute_id < n)
		output[absolute_id] = temp[block_thread_id];
}

__global__ void gather_sums(int *const array, int const *partial_sums) {
	if (blockIdx.x > 0)
		array[threadIdx.x] += partial_sums[blockIdx.x - 1];
}

void prefix_recursive(int *const output, int *const input, const int size) {
	const auto num_of_blocks = size / threads_per_block + 1;
	auto pow = 1;
	while (pow < threads_per_block)
		pow *= 2;

	GPUBuffer<int> partial_sums {num_of_blocks - 1};
	prescan<<<num_of_blocks, threads_per_block, 2 * threads_per_block>>>
		(output, input, partial_sums.data(), size);
	assert(cudaGetLastError() == cudaSuccess);
	if (num_of_blocks <= 1)
		return;

	auto cuda_status = cudaDeviceSynchronize();
	assert(cuda_status == cudaSuccess);
	prefix_recursive(partial_sums.data(), partial_sums.data(), partial_sums.size());
	gather_sums<<<num_of_blocks, threads_per_block>>>(output, partial_sums.data());
	assert(cudaGetLastError() == cudaSuccess);
}

void mapped_scan(GPUBuffer<int> &output, GPUBuffer<int> &input, const int size, const int target_value) {
	const auto num_of_blocks = size / threads_per_block + 1;
	GPUBuffer<int> mapped_input {size};
	map_to_target<<<num_of_blocks, threads_per_block>>>(mapped_input.data(), input.data(),
							    size, target_value);
	assert(cudaGetLastError() == cudaSuccess);
	prefix_recursive(output.data(), mapped_input.data(), size);
}

template<typename tuple_t>
__global__ void create_sub_batch(tuple_t *const bin,
				 const std::size_t batch_size,
				 std::size_t *const index,
				 std::size_t *const scan,
				 tuple_t *const bout,
				 const int target_node) {
	const auto id = blockIdx.x * blockDim.x + threadIdx.x;
	// No need for an explicit cycle: each GPU thread computes this in
	// parallel.
	if (id < batch_size && index[id] == target_node)
		bout[scan[index[id]] - 1] = bin[id];
}

int main(const int argc, char *const argv[]) {
	if (argc <= 1) {
		cerr << "Use as " << argv[0] << " <space-separated elements>\n";
		return -1;
	}
	constexpr auto target_value = 0;
	const auto n = argc - 1;
	GPUBuffer<int> gpu_scan {n};
	GPUBuffer<int> gpu_index {n};
	vector<int> cpu_index;
	vector<int> cpu_scan;

	cpu_scan.resize(n);
	for (auto i = 1; i < argc; ++i)
		cpu_index.push_back(std::atoi(argv[i]));

	auto cuda_status = cudaMemcpy(gpu_index.data(), cpu_index.data(),
				      n * sizeof *gpu_index.data(), cudaMemcpyHostToDevice);
	assert(cuda_status == cudaSuccess);

	mapped_scan(gpu_scan, gpu_index, n, target_value);
	cuda_status = cudaDeviceSynchronize();
	assert(cuda_status == cudaSuccess);
	cuda_status = cudaMemcpy(cpu_scan.data(), gpu_scan.data(),
				 n * sizeof *gpu_scan.data(), cudaMemcpyDeviceToHost);
	assert(cuda_status == cudaSuccess);

	cout << "Size is " << n << '\n';
	cout << "Result of scan: ";
	for (const auto &x : cpu_scan)
		cout << x << ' ';
	cout << "\n";
	std::size_t bout_size = cpu_scan[n - 1];
	GPUBuffer<std::size_t> bout {bout_size};
	return 0;
}
