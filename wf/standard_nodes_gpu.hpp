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

/**
 *  @file    standard_nodes_gpu.hpp
 *  @author  Cosimo Agati
 *  @date    12/02/2019
 *
 *  @brief Standard GPU emitter and collector used by the WindFlow library
 *
 *  @section Standard Emitter and Collector (GPU versions) (Description)
 *
 *  This file implements the standard GPU emitter and collector used by the library.
 */

#ifndef STANDARD_GPU_H
#define STANDARD_GPU_H

#include <cstddef>
#include <functional>
#include <vector>
#include <unordered_map>
#include <ff/multinode.hpp>
#include "basic.hpp"
#include "basic_emitter.hpp"
#include "gpu_utils.hpp"

namespace wf {
// TODO: Emitter should have its own stream!
/*
 * Parallel scan (prefix) implementation on GPU.  Code shamelessly adapted from
 * the book "GPU Gems 3":
 * https://developer.nvidia.com/gpugems/gpugems3/contributors
 */
// TODO: Adapt to contain target value! Right now it just performs a sum.  Idea:
// preprocess the array (easily done in parallel) to set each element to 1 if it
// equals the target value, 0 otherwise.  At this point, performing a "standard"
// parallel sum will give us precisely what we want: each result is the number
// of occurrences of target value in the subarray starting from the first
// location and ending to that position (included).
template <typename T>
__global__ void prescan(T *const g_odata, T *const g_idata, const int n,
			const T target_value) {
	extern __shared__ T temp[]; // allocated on invocation
	extern __shared__ T mapped_idata[];
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	const auto thread_id = threadIdx.x;
	auto offset = 1;

	for (auto i = index; i < n; i += stride) {
		mapped_idata[i] = g_idata[i] == target_value;
	}
	temp[2 * thread_id] = mapped_idata[2 * thread_id]; // load input into shared memory
	temp[2 * thread_id + 1] = mapped_idata[2 * thread_id + 1];
	for (auto d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
		__syncthreads();
		if (thread_id < d) {
L			auto ai = offset * (2 * thread_id + 1) - 1;
			auto bi = offset * (2 * thread_id + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thread_id == 0) {
		temp[n - 1] = 0; // clear the last element
	}
	for (auto d = 1; d < n; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (thread_id < d) {
			auto ai = offset * (2 * thread_id + 1) - 1;
			auto bi = offset * (2 * thread_id + 2) - 1;
			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thread_id] = temp[2 * thread_id]; // write results to device memory
	g_odata[2 * thread_id + 1] = temp[2 * thread_id + 1];
}

/*
 * Used to split a keyed batch to ensure keys always go to the same node.
 */
// TODO: Change names to something more self-explainatory.
template<typename tuple_t>
__global__ void create_sub_batch(tuple_t *const bin,
				 const std::size_t batch_size,
				 std::size_t *const index,
				 std::size_t *const scan,
				 tuple_t * const bout,
				 const int target_node) {
	const auto id = blockIdx.x * blockDim.x + threadIdx.x;
	// No need for an explicit cycle: each GPU thread computes this in
	// parallel.
	if (id < batch_size && index[id] == target_node) {
		bout[scan[index[id]] - 1] = bin[id];
	}
}

template<typename tuple_t>
class Standard_EmitterGPU: public Basic_Emitter {
private:
	// type of the function to map the key hashcode onto an identifier
	// starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;

	cudaStream_t cuda_stream;
	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	std::size_t *scan {nullptr};
	std::size_t destination_index;
	std::size_t num_of_destinations;
	// TODO: is it fine for a GPU emitter to be used in a Tree_Emitter?
	bool is_combined; // true if this node is used within a Tree_Emitter node
	bool have_gpu_input;
	bool have_gpu_output;

public:
	Standard_EmitterGPU(const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false)
		: routing_mode {FORWARD}, destination_index {0},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{
		if (num_of_destinations <= 0) {
			failwith("Standard_EmitterGPU initialized with non-positive number of destinations.");
		}
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		}
	}

	Standard_EmitterGPU(const routing_func_t routing_func,
			    const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false)
		: routing_mode {KEYBY}, routing_func {routing_func},
		  destination_index {0},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{
		assert(num_of_destinations);
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		}
		if (cudaMalloc(&scan, num_of_destinations * sizeof(std::size_t))) {
			failwith("Standard_EmitterGPU failed to allocate scan array.");
		}
	}

	~Standard_EmitterGPU() {
		cudaStreamDestroy(cuda_stream);
		if (scan) {
			cudaFree(scan);
		}
	}

	Basic_Emitter *clone() const {
		return new Standard_EmitterGPU<tuple_t> {*this};
	}

	int svc_init() const { return 0; }

	void *svc(void *const input) {
		if (routing_mode != KEYBY) {
			return input; // Same whether input is a tuple or batch.
		}
		if (have_gpu_input) {
			linear_keyed_gpu_partition(input);
		} else {
			auto t = reinterpret_cast<tuple_t *>(input);
			auto key = std::get<0>(t->getControlFields());
			auto hashcode = hash(key);
			destination_index = routing_func(hashcode, num_of_destinations);
			this->ff_send_out_to(t, destination_index);
			return this->GO_ON;
		}
		return nullptr; // Silence potential compiler warnings.
	}

	/*
	 * Supposed to be slow, only for performance testing purposes.
	 */
	void linear_keyed_gpu_partition(void *const input) {
		auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
		const auto raw_batch_size = handle->size * sizeof(tuple_t);
		tuple_t *cpu_tuple_buffer; // TODO: Should this be just a class member?
		if (cudaMallocHost(&cpu_tuple_buffer, raw_batch_size) != cudaSuccess) {
			failwith("Standard_EmitterGPU failed to allocate CPU buffer");
		}
		cudaMemcpy(cpu_tuple_buffer, handle->buffer, raw_batch_size, cudaMemcpyDeviceToHost);
		std::vector<std::vector<tuple_t>> sub_buffers (num_of_destinations);

		for (auto i = 0; i < handle->size; ++i) {
			const auto &tuple = cpu_tuple_buffer[i];
			const auto hashcode = hash(tuple.key) % num_of_destinations;
			sub_buffers[hashcode].push_back(tuple);
		}
		cudaFreeHost(cpu_tuple_buffer);
		for (auto i = 0; i < num_of_destinations; ++i) {
			if (sub_buffers[i].empty()) {
				continue;
			}
			const auto &cpu_sub_buffer = sub_buffers[i];
			const auto raw_size = sizeof(tuple_t) * cpu_sub_buffer.size();
			tuple_t *gpu_sub_buffer;
			if (cudaMalloc(&gpu_sub_buffer, raw_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate GPU sub-buffer");
			}
			cudaMemcpy(gpu_sub_buffer, cpu_sub_buffer.data(),
				   raw_size, cudaMemcpyHostToDevice);
			this->ff_send_out_to(new GPUBufferHandle<tuple_t>
					     {gpu_sub_buffer, cpu_sub_buffer.size()},
					     i);
		}
		cudaFree(handle->buffer);
		delete handle;
	}

	/*
	 * Actual partitioning implementation to be eventually used.
	 */
	void parallel_keyed_gpu_partition(void *const input) {
		auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
		const auto raw_batch_size = handle->size * sizeof(tuple_t);
		tuple_t *cpu_tuple_buffer; // TODO: Should this be just a class member?
		if (cudaMallocHost(&cpu_tuple_buffer, raw_batch_size) != cudaSuccess) {
			failwith("Standard_EmitterGPU failed to allocate CPU buffer");
		}
		cudaMemcpy(cpu_tuple_buffer, handle->buffer,
			   raw_batch_size, cudaMemcpyDeviceToHost);
		std::size_t cpu_hash_index[num_of_destinations]; // Stores modulo'd hash values for keys.
		for (auto i = 0; i < handle->size; ++i) {
			cpu_hash_index[i]= hash(cpu_tuple_buffer[i].key);
		}
		cudaFreeHost(cpu_tuple_buffer);
		const auto raw_index_size = sizeof(std::size_t) * num_of_destinations;
		std::size_t *gpu_hash_index;
		if (cudaMalloc(&gpu_hash_index, raw_index_size) != cudaSuccess) {
			failwith("Standard_EmitterGPU failed to allocate GPU index");
		}
		cudaMemcpy(gpu_hash_index, cpu_hash_index, raw_index_size, cudaMemcpyHostToDevice);
		for (auto i = 0; i < num_of_destinations; ++i) {
			prescan<<<1, 256, num_of_destinations, cuda_stream>>>
				(scan, gpu_hash_index, num_of_destinations, i);
			// May be inefficient, should probably start all kernels
			// in the loop in parallel.
			cudaStreamSynchronize(cuda_stream);
			const auto bout_raw_size = scan[num_of_destinations - 1] * sizeof(tuple_t);
			tuple_t *bout;
			if (cudaMalloc(&bout, bout_raw_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate partial output batch.");
			}
			create_sub_batch<<<1, 256, 0, cuda_stream>>>
				(handle->buffer, num_of_destinations,
				 gpu_hash_index, scan, bout, i);
			ff_send_out_to(new GPUBufferHandle<tuple_t>
				       {bout, scan[num_of_destinations - 1]}, i);
		}
		cudaFree(gpu_hash_index);
	}

	void svc_end() const {}

	std::size_t getNDestinations() const override { return num_of_destinations; }

	// set/unset the Tree_Emitter mode
	void setTree_EmitterMode(const bool val) override { is_combined = val; }

	// method to get a reference to the internal output queue (used in Tree_Emitter mode)
	std::vector<std::pair<void *, int>> &getOutputQueue() override {
		// TODO
	}
};

// class Standard_Collector
// FIXME: Is this needed?
class Standard_CollectorGPU: public ff::ff_minode {
public:
	Standard_CollectorGPU(const ordering_mode_t mode=TS) {
		assert(mode == TS);
	}

	void *svc(void *const t) { return t; }
};

} // namespace wf

#endif
