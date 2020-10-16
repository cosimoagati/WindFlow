/******************************************************************************
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 ******************************************************************************
 */

/*
 *  Test of the stateful batch computation (version 4). This version processes all the
 *  unique keys in the batch by using a set of workers on the GPU. A worker consists of
 *  one of more than one CUDA thread. If the number of warps on the GPU is greater or
 *  equal to the number of unique keys in the batch, only one active CUDA thread per
 *  warp is used. Otherwise, more than one active thread per warp is used. In case of
 *  a very large number of unique keys, all the threads of each warp are used. The maximum
 *  number of CUDA threads is equal to the maximum number of resident threads on the GPU.
 */

// includes
#include "common.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <unordered_map>

using namespace std;
using namespace ff;

// compute the next power of two greater than a 32-bit integer
int32_t next_power_of_two(int32_t n) {
	assert(n > 0);
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	return n + 1;
}

__device__ void process(tuple_t *t, state_t *state) {
	auto &index           = state->index;
	auto &buffer_location = (state->buffer)[index];
	auto &value           = t->value;

	if (value % 2 == 0) { // even
		if (value % 4 == 0) {
			buffer_location += value;
			value = buffer_location;
			index = (index + 1) % 100;
		} else {
			buffer_location *= value;
			value = buffer_location;
			index = (index + 2) % 100;
		}
	} else { // odd
		if (value % 5 == 0) {
			buffer_location *= value;
			value = buffer_location;
			index = (index + 3) % 100;
		} else {
			buffer_location += value;
			value = buffer_location;
			index = (index + 4) % 100;
		}
	}
}

// CUDA Kernel Stateful_Processing_Kernel
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, size_t *keys, size_t *dist_keys_gpu,
                                           size_t num_dist_keys, state_t **states, size_t len,
                                           int num_active_thread_per_warp) {
	const int id = threadIdx.x + blockIdx.x * blockDim.x; // id of the thread in the kernel
	const int threads_per_worker =
	        warpSize / num_active_thread_per_warp;  // number of threads composing a worker entity
	const int num_threads = gridDim.x * blockDim.x; // number of threads in the kernel
	const int num_workers = num_threads / threads_per_worker; // number of workers
	const int id_worker   = id / threads_per_worker;          // id of the worker
	// only the first thread of each warp works, the others are idle
	if (id % threads_per_worker == 0) {
		for (size_t id_key = id_worker; id_key < num_dist_keys; id_key += num_workers) {
			const auto key = dist_keys_gpu[id_key]; // key used
			for (size_t i = 0; i < len; i++) {
				if (key == keys[i]) {
					process(&(tuples[i]), states[i]);
				}
			}
		}
	}
}

class Worker : public ff_node_t<batch_t> {
public:
	unsigned long                    received_batch = 0;
	size_t                           received       = 0; // counter of received tuples
	size_t                           par_deg;
	size_t                           id;
	cudaStream_t                     cudaStream;
	unordered_map<size_t, state_t *> hashmap;
	unsigned long                    tot_elapsed_nsec   = 0;
	int                              numSMs             = 0;
	int                              max_threads_per_sm = 0;
	int                              max_blocks_per_sm  = 0;
	int                              threads_per_warp   = 0;

	Worker(size_t _par_deg, size_t _id) : par_deg(_par_deg), id(_id) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
		// get some configuration parameters of the GPU
		gpuErrChk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
		gpuErrChk(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
		                                 0));
		gpuErrChk(
		        cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
		gpuErrChk(cudaDeviceGetAttribute(&threads_per_warp, cudaDevAttrWarpSize, 0));
		assert(numSMs > 0);             // 15
		assert(max_threads_per_sm > 0); // 2048
		assert(max_blocks_per_sm > 0);  // 16
		assert(threads_per_warp > 0);   // 32
		if (id == 0) {
			cout << "NumSMs: " << numSMs << ", threads per SM: " << max_threads_per_sm
			     << ", blocks per SM: " << max_blocks_per_sm << ", threads per warp "
			     << threads_per_warp << endl;
		}
	}

	~Worker() { gpuErrChk(cudaStreamDestroy(cudaStream)); }

	batch_t *svc(batch_t *const b) {
		volatile unsigned long start_time_nsec = current_time_nsecs();
		received_batch++;
		received += b->size;
		size_t *keys_cpu = nullptr;
		cudaMallocHost(&keys_cpu, sizeof(size_t) * b->size);
		// copy keys_gpu to keys_cpu
		gpuErrChk(cudaMemcpyAsync(keys_cpu, b->keys_gpu, b->size * sizeof(size_t),
		                          cudaMemcpyDeviceToHost, cudaStream));
		gpuErrChk(cudaStreamSynchronize(cudaStream));
#if 1
		// check the correctness of the received batch
		// std::cout << "Worker " << id << ", Keys [ ";
		for (size_t i = 0; i < b->size; i++) {
			assert(keys_cpu[i] % par_deg == id);
			// std::cout << keys_cpu[i] << " ";
		}
		// std::cout << "]" << std::endl;
#endif
		// adds and lookups in the hashmap
		const auto state_ptrs_cpu = (state_t **) malloc(b->size * sizeof(state_t *));
		int        num_dist_keys  = 0;
		unordered_map<size_t, size_t> dist_map;
		const auto                    dist_keys_cpu = (size_t *) malloc(b->size * sizeof(size_t));
		for (size_t i = 0; i < b->size; i++) {
			const auto key = keys_cpu[i];
			auto       it  = hashmap.find(key);
			if (it == hashmap.end()) {
				// create the state of that key
				state_t *state = nullptr;
				cudaMalloc(&state, sizeof(state_t));
				cudaMemsetAsync(state, 0, sizeof(state_t), cudaStream);
				hashmap.insert(std::make_pair(key, state));
				it = hashmap.find(key);
			}
			state_ptrs_cpu[i] = (*it).second;
			// count distinct keys in the batch
			const auto it2 = dist_map.find(key);
			if (it2 == dist_map.end()) {
				dist_map.insert(std::make_pair(key, 1));
				dist_keys_cpu[num_dist_keys] = key;
				num_dist_keys++;
			}
		}
		// copy state_ptrs_cpu to state_ptrs_gpu
		state_t **state_ptrs_gpu = nullptr;
		cudaMalloc(&state_ptrs_gpu, b->size * sizeof(state_t *));
		gpuErrChk(cudaMemcpyAsync(state_ptrs_gpu, state_ptrs_cpu, b->size * sizeof(state_t *),
		                          cudaMemcpyHostToDevice, cudaStream));
		// copy distinct keys on GPU
		size_t *dist_keys_gpu = nullptr;
		cudaMalloc(&dist_keys_gpu, sizeof(size_t) * num_dist_keys);
		gpuErrChk(cudaMemcpyAsync(dist_keys_gpu, dist_keys_cpu, num_dist_keys * sizeof(size_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		// launch the kernel to compute the results
		const int warps_per_block = ((max_threads_per_sm / max_blocks_per_sm) / threads_per_warp);
		const int tot_num_warps   = warps_per_block * max_blocks_per_sm * numSMs;
		// compute how many threads should be active per warps
		int32_t x = (int32_t) ceil(((double) num_dist_keys) / tot_num_warps);
		if (x > 1) {
			x = next_power_of_two(x);
		}
		const int num_active_thread_per_warp = std::min(x, threads_per_warp);
		const int num_blocks = std::min((int) ceil(((double) num_dist_keys) / warps_per_block),
		                                numSMs * max_blocks_per_sm);
		Stateful_Processing_Kernel<<<num_blocks, warps_per_block * threads_per_warp, 0, cudaStream>>>(
		        b->data_gpu, b->keys_gpu, dist_keys_gpu, num_dist_keys, state_ptrs_gpu, b->size,
		        num_active_thread_per_warp);
		free(state_ptrs_cpu);
		free(dist_keys_cpu);
		cudaFree(state_ptrs_gpu);
		cudaFree(dist_keys_gpu);
		cudaFreeHost(keys_cpu);
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return b;
	}

	void svc_end() {
		printf("[Worker] average service time: %f usec\n",
		       (((double) tot_elapsed_nsec) / received_batch) / 1000);
		// std::cout << "[Worker] average service time: " << (((double)
		// tot_elapsed_nsec)/received_batch) / 1000 << " usec" << std::endl;
	}
};

int main(int argc, char *argv[]) { return run_test<Worker>(argc, argv); }
