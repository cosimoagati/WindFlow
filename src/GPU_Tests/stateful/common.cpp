#include "common.hpp"
#include "../zipf.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <unordered_map>

using namespace std;
using namespace ff;

key_less_than_op::key_less_than_op(const size_t _n_dest) : n_dest(_n_dest) {}

__host__ __device__ bool key_less_than_op::operator()(const size_t k1, const size_t k2) {
	return ((k1 % n_dest) < (k2 % n_dest));
}

key_equal_to_op::key_equal_to_op(const size_t _n_dest) : n_dest(_n_dest) {}

__host__ __device__ bool key_equal_to_op::operator()(const size_t k1, const size_t k2) {
	return ((k1 % n_dest) == (k2 % n_dest));
}

__host__ __device__ tuple_t::tuple_t() : key(0), value(0) {}

tuple_t::tuple_t(const size_t _key, const int64_t _value) : key(_key), value(_value) {}

batch_t::batch_t(const size_t _size) : size(_size) {
	cudaMalloc(&data_gpu, size * sizeof(tuple_t));
	cudaMalloc(&keys_gpu, size * sizeof(size_t));
	raw_data_gpu = data_gpu;
	raw_keys_gpu = keys_gpu;
}

batch_t::batch_t(const size_t _size, tuple_t *const _data_gpu, size_t *const _keys_gpu, const size_t _offset,
                 atomic<size_t> *const _counter) {
	size         = _size;
	raw_data_gpu = _data_gpu;
	raw_keys_gpu = _keys_gpu;
	data_gpu     = raw_data_gpu + _offset;
	keys_gpu     = raw_keys_gpu + _offset;
	counter      = _counter;
}

// destructor
batch_t::~batch_t() {
	if (counter != nullptr) {
		size_t old_cnt = counter->fetch_sub(1);
		if (old_cnt == 1) {
			if (cleanup) {
				cudaFree(raw_data_gpu);
				cudaFree(raw_keys_gpu);
			}
			delete counter;
		}
	} else if (cleanup) {
		cudaFree(raw_data_gpu);
		cudaFree(raw_keys_gpu);
	}
}

__host__ __device__ state_t::state_t() {
	index = 0;
	memset(buffer, 0, sizeof(int) * 100);
}

// constructor
Emitter::Emitter(const size_t _n_dest, const size_t _batch_len) : n_dest(_n_dest), batch_len(_batch_len) {
	// allocate unique_keys_cpu array
	cudaMallocHost(&unique_dests_cpu, sizeof(size_t) * n_dest);
	// allocate freqs_keys_cpu array
	cudaMallocHost(&freqs_dests_cpu, sizeof(int) * n_dest);
	// initialize CUDA stream
	gpuErrChk(cudaStreamCreate(&cudaStream));
}

Emitter::~Emitter() {
	cudaFreeHost(unique_dests_cpu);
	cudaFreeHost(freqs_dests_cpu);
	gpuErrChk(cudaStreamDestroy(cudaStream));
}

// svc method
batch_t *Emitter::svc(batch_t *const b) {
	volatile unsigned long start_time_nsec = current_time_nsecs();
	received++;
	if (n_dest == 1) {
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		const auto             elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return b;
	}
	// sort of the input batch: inputs directed to the same destination are placed contiguously in
	// the batch
	const auto th_data_gpu = thrust::device_pointer_cast(b->data_gpu);
	const auto th_keys_gpu = thrust::device_pointer_cast(b->keys_gpu);
	thrust::sort_by_key(thrust::cuda::par.on(cudaStream), th_keys_gpu, th_keys_gpu + b->size, th_data_gpu,
	                    key_less_than_op(n_dest));

	// compute the number of unique occurrences of each destination identifier
	thrust::device_vector<int> ones_gpu(b->size); // for sure they are not more than n_dest
	thrust::fill(ones_gpu.begin(), ones_gpu.end(), 1);
	thrust::device_vector<size_t> unique_dests_gpu(n_dest); // for sure they are not more than n_dest
	thrust::device_vector<int>    freqs_dests_gpu(n_dest);  // for sure they are not more than n_dest

	const auto end = thrust::reduce_by_key(
	        thrust::cuda::par.on(cudaStream), th_keys_gpu, th_keys_gpu + b->size, ones_gpu.begin(),
	        unique_dests_gpu.begin(), freqs_dests_gpu.begin(), key_equal_to_op(n_dest));
	const size_t num_found_dests = end.first - unique_dests_gpu.begin();
	assert(num_found_dests > 0);
	gpuErrChk(cudaMemcpyAsync(unique_dests_cpu, thrust::raw_pointer_cast(unique_dests_gpu.data()),
	                          sizeof(size_t) * num_found_dests, cudaMemcpyDeviceToHost, cudaStream));
	gpuErrChk(cudaMemcpyAsync(freqs_dests_cpu, thrust::raw_pointer_cast(freqs_dests_gpu.data()),
	                          sizeof(int) * num_found_dests, cudaMemcpyDeviceToHost, cudaStream));
	gpuErrChk(cudaStreamSynchronize(cudaStream));
	size_t offset  = 0;
	auto   counter = new atomic<size_t>(
                num_found_dests); // to deallocate correctly the GPU buffer within the batch
	// for each destination that must receive data
	for (size_t i = 0; i < num_found_dests; i++) {
		const auto result = freqs_dests_cpu[i];
		const auto bout   = new batch_t(result, b->data_gpu, b->keys_gpu, offset, counter);
		this->ff_send_out_to(bout, unique_dests_cpu[i] % n_dest);
		offset += result;
	}
	b->cleanup = false; // we don't want to deallocate the input GPU buffer because it is used by
	                    // the transmitted batches
	delete b;
	volatile unsigned long end_time_nsec     = current_time_nsecs();
	unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
	tot_elapsed_nsec += elapsed_time_nsec;
	return this->GO_ON;
}

void Emitter::svc_end() {
	cout << "[Emitter] average service time: " << (((double) tot_elapsed_nsec) / received) / 1000
	     << " usec" << endl;
}

Sink::Sink() : start_time_us(current_time_usecs()), last_time_us(current_time_usecs()) {
	// initialize CUDA stream
	gpuErrChk(cudaStreamCreate(&cudaStream));
}

Sink::~Sink() { gpuErrChk(cudaStreamDestroy(cudaStream)); }

batch_t *Sink::svc(batch_t *const b) {
	received += b->size;
	received_per_sample += b->size;
	tuple_t *data_cpu = nullptr;
	cudaMallocHost(&data_cpu, sizeof(tuple_t) * b->size);
	gpuErrChk(cudaMemcpyAsync(data_cpu, b->data_gpu, b->size * sizeof(tuple_t), cudaMemcpyDeviceToHost,
	                          cudaStream));
	gpuErrChk(cudaStreamSynchronize(cudaStream));
	for (size_t i = 0; i < b->size; i++)
		correctness += data_cpu[i].value;

	// print the throughput per second
	if (current_time_usecs() - last_time_us > 1000000) {
		double elapsed_sec = ((double) (current_time_usecs() - start_time_us)) / 1000000;
		std::cout << "[SINK] time " << elapsed_sec << " received " << received_per_sample
		          << std::endl;
		received_per_sample = 0;
		last_time_us        = current_time_usecs();
	}
	delete b;
	cudaFreeHost(data_cpu);
	return this->GO_ON;
}

void Sink::svc_end() {
	double elapsed_sec = ((double) (current_time_usecs() - start_time_us)) / 1000000;
	// std::cout << "[SINK] time " << elapsed_sec << " received " << received_per_sample << std::endl;
	// std::cout << "[SINK] total received " << received << std::endl;
	printf("[SINK] time %f received %d\n", elapsed_sec, received_per_sample);
	printf("[SINK] total received %d with total count %ld\n", received, correctness);
}
