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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>
#include <unordered_map>
#include <ff/multinode.hpp>
#include "basic.hpp"
#include "basic_emitter.hpp"
#include "gpu_utils.hpp"

// # define PARALLEL_PARTITION

namespace wf {
// Use int instead of std::size_t since it's smaller and faster.  Beware
// truncating!
template<typename T>
__global__ void prescan(T *const g_odata, T *const g_idata, const T n,
			const T target_value, const T power_of_two) {
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

template<typename T>
constexpr T get_closest_power_of_two(const T n) {
	auto i = 1;
	while (i < n) {
		i *= 2;
	}
	return i;
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
				 tuple_t *const bout,
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
	using buffer_handle_t = GPUBufferHandle<tuple_t>;

	cudaStream_t cuda_stream;
	cudaError_t cuda_error; // Used to store and catch CUDA errors;
	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	PinnedCPUBuffer<tuple_t> cpu_tuple_buffer;
	PinnedCPUBuffer<std::size_t> cpu_hash_index;
	GPUBuffer<std::size_t> gpu_hash_index; // Stores modulo'd hash values for keys.
	GPUBuffer<std::size_t> scan;

	std::size_t destination_index {0};
	std::size_t num_of_destinations;
	// std::size_t current_allocated_size;
	int gpu_blocks;
	int gpu_threads_per_block;

	// TODO: is it fine for a GPU emitter to be used in a Tree_Emitter?
	bool is_combined; // true if this node is used within a Tree_Emitter node
	bool have_gpu_input;
	bool have_gpu_output;

public:
	Standard_EmitterGPU(const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false,
			    const int gpu_threads_per_block=256)
		: routing_mode {FORWARD},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(num_of_destinations
					/ static_cast<float>(gpu_threads_per_block))}
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
			    const bool have_gpu_output=false,
			    const int gpu_threads_per_block=256)
		: routing_mode {KEYBY}, routing_func {routing_func},
		  num_of_destinations {num_of_destinations},
		  cpu_tuple_buffer (num_of_destinations),
		  cpu_hash_index (num_of_destinations),
		  gpu_hash_index {num_of_destinations},
		  scan {num_of_destinations}, have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(num_of_destinations
					/ static_cast<float>(gpu_threads_per_block))}
	{
		if (num_of_destinations <= 0) {
			failwith("Standard_EmitterGPU initialized with non-positive number of destinations.");
		}
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		}
	}

	~Standard_EmitterGPU() {
		cuda_error = cudaStreamDestroy(cuda_stream);
		assert(cuda_error == cudaSuccess);
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
			const auto handle = reinterpret_cast<buffer_handle_t *>(input);
#ifdef PARALLEL_PARTITION
			parallel_keyed_gpu_partition(handle);
#else
			linear_keyed_gpu_partition(handle);
#endif
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			const auto key = std::get<0>(t->getControlFields());
			const auto hashcode = hash(key);
			destination_index = routing_func(hashcode, num_of_destinations);
			this->ff_send_out_to(t, destination_index);
			return this->GO_ON;
		}
		return nullptr; // Silence potential compiler warnings.
	}

	/*
	 * Supposed to be slow, only for performance testing purposes.
	 */
	void linear_keyed_gpu_partition(buffer_handle_t *const handle) {
		const auto raw_batch_size = handle->size * sizeof *handle->buffer;
		cpu_tuple_buffer.enlarge(handle->size);
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle->buffer,
					     raw_batch_size, cudaMemcpyDeviceToHost, cuda_stream);
		assert(cuda_error == cudaSuccess);
		std::vector<std::vector<tuple_t>> sub_buffers (num_of_destinations);

		cuda_error = cudaStreamSynchronize(cuda_stream);
		assert(cuda_error == cudaSuccess);
		for (auto i = 0; i < handle->size; ++i) {
			const auto &tuple = cpu_tuple_buffer[i];
			const auto key = std::get<0>(tuple.getControlFields());
			const auto hashcode = hash(key) % num_of_destinations;
			sub_buffers[hashcode].push_back(tuple);
		}
		for (auto i = 0; i < num_of_destinations; ++i) {
			if (sub_buffers[i].empty()) {
				continue;
			}
			const auto &cpu_sub_buffer = sub_buffers[i];
			const auto raw_size = sizeof *cpu_sub_buffer.data() * cpu_sub_buffer.size();
			tuple_t *gpu_sub_buffer;
			if (cudaMalloc(&gpu_sub_buffer, raw_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate GPU sub-buffer");
			}
			cuda_error = cudaMemcpyAsync(gpu_sub_buffer, cpu_sub_buffer.data(),
						     raw_size, cudaMemcpyHostToDevice, cuda_stream);
			assert(cuda_error == cudaSuccess);

			cuda_error = cudaStreamSynchronize(cuda_stream);
			assert(cuda_error == cudaSuccess);
			this->ff_send_out_to(new buffer_handle_t
					     {gpu_sub_buffer, cpu_sub_buffer.size()},
					     i);
		}
		cuda_error = cudaFree(handle->buffer);
		assert(cuda_error == cudaSuccess);
		delete handle;
	}

	/*
	 * Actual partitioning implementation to be eventually used.
	 */
	void parallel_keyed_gpu_partition(buffer_handle_t *const handle) {
		cpu_tuple_buffer.enlarge(handle->size);
		cpu_hash_index.enlarge(handle->size);
		gpu_hash_index.enlarge(handle->size);
		scan.enlarge(handle->size);

		const auto raw_batch_size = handle->size * sizeof *handle->buffer;
		// TODO: Can we avoid this copy on cpu?
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle->buffer, raw_batch_size,
					     cudaMemcpyDeviceToHost, cuda_stream);
		assert(cuda_error == cudaSuccess);
		cudaStreamSynchronize(cuda_stream);
		for (auto i = 0; i < handle->size; ++i) {
			const auto key = std::get<0>(cpu_tuple_buffer[i].getControlFields());
			cpu_hash_index[i] = hash(key) % num_of_destinations;
		}
		const auto raw_index_size = sizeof *cpu_hash_index.data() * handle->size;
		cuda_error = cudaMemcpyAsync(gpu_hash_index.data(), cpu_hash_index.data(),
					     raw_index_size, cudaMemcpyHostToDevice, cuda_stream);
		assert(cuda_error == cudaSuccess);

		const auto pow = get_closest_power_of_two(handle->size);
		for (auto i = 0; i < num_of_destinations; ++i) {
			prescan<<<gpu_blocks, gpu_threads_per_block,
				2 * pow * sizeof(int), cuda_stream>>>
				(scan.data(), gpu_hash_index.data(), handle->size,
				 static_cast<std::size_t>(i), pow);
			assert(cudaGetLastError() == cudaSuccess);

			// Used for debugging.
			// std::size_t cpu_scan[handle->size];
			// cudaStreamSynchronize(cuda_stream);
			// cudaMemcpy(cpu_scan, scan, handle->size * sizeof *cpu_scan,
			// 	   cudaMemcpyDeviceToHost);

			std::size_t bout_size;
			cuda_error = cudaMemcpyAsync(&bout_size, &scan[handle->size - 1],
						     sizeof bout_size, cudaMemcpyDeviceToHost,
						     cuda_stream);
			assert(cuda_error == cudaSuccess);
			cuda_error = cudaStreamSynchronize(cuda_stream);
			assert(cuda_error == cudaSuccess);

			const auto bout_raw_size = bout_size * sizeof(tuple_t);
			tuple_t *bout;
			if (cudaMalloc(&bout, bout_raw_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate partial output batch.");
			}
			create_sub_batch<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
				(handle->buffer, num_of_destinations,
				 gpu_hash_index.data(), scan.data(), bout, i);
			assert(cudaGetLastError() == cudaSuccess);
			cuda_error = cudaStreamSynchronize(cuda_stream);
			assert(cuda_error == cudaSuccess);
			ff_send_out_to(new buffer_handle_t {bout, bout_size}, i);
		}
		cuda_error = cudaFree(handle->buffer);
		assert(cuda_error);
		delete handle;
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
