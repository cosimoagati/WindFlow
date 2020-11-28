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
 *  Test of the stateful batch computation (version 1). This version uses one
 *  CUDA thread per input in the batch. Each CUDA thread scans the whole batch
 *  and executes all the inputs whose keys are assigned to it.
 */

// includes
#include "../zipf.hpp"
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

// processing function on a tuple and the state of its key
__device__ void process(tuple_t *const t, state_t *const state) {
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
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, size_t *keys, state_t **states, size_t len) {
	const int id          = threadIdx.x + blockIdx.x * blockDim.x; // id of the thread in the kernel
	const int num_threads = gridDim.x * blockDim.x;                // number of threads in the kernel
	for (size_t i = 0; i < len; i++) {
		if (keys[i] % num_threads == id)
			process(&(tuples[i]), states[i]);
	}
}

// Worker class
class Worker : public ff_node_t<batch_t> {
public:
	unsigned long                    received_batch = 0;
	size_t                           received       = 0; // counter of received tuples
	size_t                           par_deg;
	size_t                           id;
	cudaStream_t                     cudaStream;
	unordered_map<size_t, state_t *> hashmap;
	unsigned long                    tot_elapsed_nsec = 0;

	// constructor
	Worker(size_t _par_deg, size_t _id) : par_deg(_par_deg), id(_id) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	// destructor
	~Worker() { gpuErrChk(cudaStreamDestroy(cudaStream)); }

	// svc method
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
		// for each key in the batch create the state if not present
		auto state_ptrs_cpu = (state_t **) malloc(b->size * sizeof(state_t *));
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
		}
		// copy state_ptrs_cpu to state_ptrs_gpu
		state_t **state_ptrs_gpu = nullptr;
		cudaMalloc(&state_ptrs_gpu, b->size * sizeof(state_t *));
		gpuErrChk(cudaMemcpyAsync(state_ptrs_gpu, state_ptrs_cpu, b->size * sizeof(state_t *),
		                          cudaMemcpyHostToDevice, cudaStream));
		// proces the batch in a stateful manner
		auto num_blocks = (size_t) ceil(((double) b->size)
		                                / threads_per_block); // it can be too large, no check here...
		Stateful_Processing_Kernel<<<num_blocks, threads_per_block, 0, cudaStream>>>(
		        b->data_gpu, b->keys_gpu, state_ptrs_gpu, b->size);

		// deallocate buffers on CPU and GPU
		free(state_ptrs_cpu);
		cudaFree(state_ptrs_gpu);
		cudaFreeHost(keys_cpu);
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return b;
	}

	// svc_end method
	void svc_end() {
		printf("[Worker] average service time: %f usec\n",
		       (((double) tot_elapsed_nsec) / received_batch) / 1000);
		// std::cout << "[Worker] average service time: " << (((double)
		// tot_elapsed_nsec)/received_batch) / 1000 << " usec" << std::endl;
	}
};

int main(int argc, char *argv[]) {
	return run_test<Worker, shifted_zipf_distribution<std::size_t>>(argc, argv);
}
