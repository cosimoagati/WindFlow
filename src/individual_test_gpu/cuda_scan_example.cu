#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

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
	prescan<<<1, 8, 2 * n>>>(gpu_scan, gpu_index, n, target_value);
	cudaMemcpy(cpu_scan.data(), gpu_scan, n * sizeof *gpu_scan, cudaMemcpyDeviceToHost);
	cout << "Result of scan: "
	for (const auto &x : cpu_scan) {
		cout << x << ' ';
	}
	cout << "\n";
	return 0;
}
