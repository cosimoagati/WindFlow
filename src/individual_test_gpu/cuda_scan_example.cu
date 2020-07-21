#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

__global__ void prescan(int *const g_odata, int *const g_idata, const int n,
			const int target_value, const int power_of_two) {
	// extern __shared__ int temp[]; // allocated on invocation
	// int *const mapped_idata = temp + n;
	extern __shared__ int mapped_idata[];
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	// const auto thread_id = threadIdx.x;
	const auto thread_id = threadIdx.x; // Assumes a single block.

	for (auto i = index; i < n; i += stride) {
		mapped_idata[i] = g_idata[i] == target_value;
	}
	// TODO: zero-out the rest of the array?
	for (auto i = n; i < power_of_two; i += stride) {
		mapped_idata[i] = 0;
	}
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= power_of_two; stride *= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if (index < 2 * power_of_two) {
			mapped_idata[index] += mapped_idata[index - stride];
		}
	}
	for (auto stride = power_of_two / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (thread_id + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * power_of_two) {
			mapped_idata[index + stride] += mapped_idata[index];
		}
	}
	// Simple, but probably inefficient and redundant: write data to output.
	__syncthreads();
	for (auto i = index; i < n; i += stride) {
		g_odata[i] = mapped_idata[i];
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

int get_closest_power_of_two(const int n) {
	auto i = 1;
	while (i < n) {
		i *= 2;
	}
	return i;
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
	int *gpu_index;
	int *gpu_scan;

	cpu_scan.resize(n);
	for (auto i = 1; i < argc; ++i) {
		cpu_index.push_back(std::atoi(argv[i]));
	}
	if (cudaMalloc(&gpu_scan, n * sizeof *gpu_scan) != cudaSuccess) {
		cerr << "Error allocating gpu_scan.\n";
		return -1;
	}
	if (cudaMalloc(&gpu_index, n * sizeof *gpu_index) != cudaSuccess) {
		cerr << "Error allocating gpu_index.\n";
		return -1;
	}
	cudaMemcpy(gpu_index, cpu_index.data(), n * sizeof *gpu_index, cudaMemcpyHostToDevice);
	const auto pow = get_closest_power_of_two(n);
	prescan<<<1, 256, 2 * pow>>>(gpu_scan, gpu_index, n, target_value, pow);
	cudaDeviceSynchronize();
	cudaMemcpy(cpu_scan.data(), gpu_scan, n * sizeof *gpu_scan, cudaMemcpyDeviceToHost);
	cout << "Size is " << n << '\n';
	cout << "Result of scan: ";
	for (const auto &x : cpu_scan) {
		cout << x << ' ';
	}
	cout << "\n";
	std::size_t bout_size = cpu_scan[n - 1];
	const auto bout_raw_size = bout_size * sizeof(int);
	int *bout;
	if (cudaMalloc(&bout, bout_raw_size) != cudaSuccess) {
		cerr << "Failed to allocate partial output batch.";
		return -1;
	}
	return 0;
}
