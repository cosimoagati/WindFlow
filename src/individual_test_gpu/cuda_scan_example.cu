#include <cstdlib>
#include <iostream>
#include <vector>
#include "../../wf/gpu_utils.hpp"

using namespace std;

__global__ void prescan_large(int *const output, int *const input,
			      int *const sums, const int end,
			      const int target_value,
			      const int power_of_two) {
	extern __shared__ int mapped_idata[];
	const auto absolute_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	const auto block_thread_id = threadIdx.x;

	for (auto i = block_thread_id; i < power_of_two; i += stride) {
		mapped_idata[i] = input[i] == target_value;
	}
	// TODO: zero-out the rest of the array?
	// if (absolute_thread_id / power_of_two < 1) {
	// 	for (auto i = block_thread_id; i < power_of_two; i += stride) {
	// 		mapped_idata[i] = 0;
	// 	}
	// }
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= power_of_two; stride *= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index < 2 * power_of_two) {
			mapped_idata[index] += mapped_idata[index - stride];
		}
	}
	for (auto stride = power_of_two / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * power_of_two) {
			mapped_idata[index + stride] += mapped_idata[index];
		}
	}
	__syncthreads();
	if (block_thread_id == 0) {
		sums[blockIdx.x] = mapped_idata[end - 1];
	}
	__syncthreads();
	// Perform a parallel prefix sum on the partial sum array.
	if (blockIdx.x == 0) {
		for (auto stride = 1; stride <= power_of_two; stride *= 2) {
			__syncthreads();
			const auto index = (block_thread_id + 1) * stride * 2 - 1;
			if (index < 2 * power_of_two) {
				sums[index] += sums[index - stride];
			}
		}
		for (auto stride = power_of_two / 2; stride > 0; stride /= 2) {
			__syncthreads();
			const auto index = (block_thread_id + 1) * stride * 2 - 1;
			if ((index + stride) < 2 * power_of_two) {
				sums[index + stride] += sums[index];
			}
		}
		// Sum partial values to the sum of the preceding block.
		if (blockIdx.x > 0) {
			for (auto i = block_thread_id; i < power_of_two; i += stride) {
				mapped_idata[i] += sums[blockIdx.x - 1];
			}
		}
	}
	// Simple, but probably inefficient and redundant: write data to output.
	__syncthreads();
	for (auto i = absolute_thread_id; i < end; i += stride) {
		output[i] = mapped_idata[i];
	}
}

__global__ void unmapped_scan(int *const )


__global__ void prescan(int *const g_odata, int *const g_idata,
			const int n, const int target_value, const int pow) {
	extern __shared__ int mapped_idata[];
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	// const auto thread_id = threadIdx.x;
	const auto thread_id = threadIdx.x; // Assumes a single block.

	for (auto i = index; i < n; i += stride) {
		mapped_idata[i] = g_idata[i] == target_value;
	}
	// TODO: zero-out the rest of the array? Is this necessary?
	// for (auto i = n; i < pow; i += stride) {
	// 	mapped_idata[i] = 0;
	// }
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= pow; stride *= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if (index < 2 * pow) {
			mapped_idata[index] += mapped_idata[index - stride];
		}
	}
	for (auto stride = pow / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * pow) {
			mapped_idata[index + stride] += mapped_idata[index];
		}
	}
	// Simple, but probably inefficient and redundant: write data to output.
	__syncthreads();
	for (auto i = index; i < n; i += stride) {
		g_odata[i] = mapped_idata[i];
	}
}

void mapped_scan(int *const output, int *const input, const int size,
		 const int target_value) {
	static constexpr auto threads_per_block = 512;
	auto pow = 1;
	if (size <= threads_per_block) {
		while (pow < size) { // Get closest power of two above or equal to size.
			pow *= 2;
		}
		prescan<<<1, pow, 2 * pow * sizeof(int)>>>(output, input, size, target_value, pow);
	} else {
		const auto num_of_blocks = size / threads_per_block + 1;
		while (pow < threads_per_block) {
			pow *= 2;
		}
		GPUBuffer<int> partial_sums {num_of_blocks - 1};
		prescan_large<<<num_of_blocks, 2 * pow * sizeof(int)>>>
			(output, input, partial_sums.data(), size, target_value, pow);
		int cpu_partial_sums[num_of_blocks - 1];
		cudaMemcpy(cpu_partial_sums, partial_sums.data(), partial_sums.size() * sizeof(int). cudaMemcpyDeviceToHost);
		cout << "a" << endl;
	}
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
	if (id < batch_size && index[id] == target_node) {
		bout[scan[index[id]] - 1] = bin[id];
	}
}

int main(const int argc, char *const argv[]) {
	if (argc <= 1) {
		cerr << "Use as " << argv[0] << " <space-separated elements>\n";
		return -1;
	}
	constexpr auto target_value = 0;
	const auto n = argc - 1;
	vector<int> cpu_index;
	vector<int> cpu_scan;
	cpu_scan.resize(n);
	for (auto i = 1; i < argc; ++i) {
		cpu_index.push_back(std::atoi(argv[i]));
	}
	GPUBuffer<int> gpu_scan {n};
	GPUBuffer<int> gpu_index {n};

	cudaMemcpy(gpu_index.data(), cpu_index.data(),
		   n * sizeof *gpu_index.data(), cudaMemcpyHostToDevice);
	mapped_scan(gpu_scan.data(), gpu_index.data(), n, target_value);
	cudaDeviceSynchronize();
	cudaMemcpy(cpu_scan.data(), gpu_scan.data(),
		   n * sizeof *gpu_scan.data(), cudaMemcpyDeviceToHost);
	cout << "Size is " << n << '\n';
	cout << "Result of scan: ";
	for (const auto &x : cpu_scan) {
		cout << x << ' ';
	}
	cout << "\n";
	std::size_t bout_size = cpu_scan[n - 1];
	GPUBuffer<std::size_t> bout {bout_size};
	return 0;
}
