#include <cstdlib>
#include <iostream>
#include <vector>
#include "../../wf/gpu_utils.hpp"

using namespace std;
using wf::GPUBuffer;
using wf::GPUStream;
using wf::mapped_scan;

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
	GPUStream stream;
	vector<int> cpu_index;
	vector<int> cpu_scan;


	cpu_scan.resize(n);
	for (auto i = 1; i < argc; ++i)
		cpu_index.push_back(std::atoi(argv[i]));

	auto cuda_status = cudaMemcpy(gpu_index.data(), cpu_index.data(),
				      n * sizeof *gpu_index.data(), cudaMemcpyHostToDevice);
	assert(cuda_status == cudaSuccess);

	mapped_scan(gpu_scan, gpu_index, n, target_value, stream);
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
