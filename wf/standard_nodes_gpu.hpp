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
#include <utility>

#include <ff/multinode.hpp>
#include "basic.hpp"
#include "basic_emitter.hpp"
#include "gpu_utils.hpp"

// # define PARALLEL_PARTITION

namespace wf {
constexpr auto threads_per_block = 512;

// Use int instead of std::size_t since it's smaller and faster.  Beware
// truncating!
__global__ void map_to_target(int *const output, int *const input,
			      const int n, const int target_value) {
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	if (absolute_id < n) {
		output[absolute_id] = input[absolute_id] == target_value;
	}
}

__global__ void prescan(int *const output, int *const input,
			int *const partial_sums, const int n) {
	extern __shared__ int temp[];
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	const auto block_thread_id = threadIdx.x;

	if (absolute_id < n) {
		temp[block_thread_id] = input[absolute_id];
	}
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= n; stride *= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index < 2 * n) {
			temp[index] += temp[index - stride];
		}
	}
	for (auto stride = n / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * n) {
			temp[index + stride] += temp[index];
		}
	}
	__syncthreads();
	if (block_thread_id == 0 && blockIdx.x > 0) {
		partial_sums[blockIdx.x - 1] = temp[n - 1];
	}
	// Simple, but probably inefficient and redundant: write data to output.
	if (absolute_id < n) {
		output[absolute_id] = temp[block_thread_id];
	}
}

__global__ void gather_sums(int *const array, int const *partial_sums) {
	if (blockIdx.x > 0) {
		array[threadIdx.x] += partial_sums[blockIdx.x - 1];
	}
}

void prefix_recursive(int *const output, int *const input, const int size) {
	const auto num_of_blocks = size / threads_per_block + 1;
	auto pow = 1;
	while (pow < threads_per_block) {
		pow *= 2;
	}
	GPUBuffer<int> partial_sums {num_of_blocks - 1};
	prescan<<<num_of_blocks, threads_per_block, 2 * threads_per_block>>>
		(output, input, partial_sums.data(), size);
	assert(cudaGetLastError() == cudaSuccess);
	if (num_of_blocks <= 1) {
		return;
	}
	auto cuda_status = cudaDeviceSynchronize();
	assert(cuda_status == cudaSuccess);
	prefix_recursive(partial_sums.data(), partial_sums.data(), partial_sums.size());
	gather_sums<<<num_of_blocks, threads_per_block>>>(output, partial_sums.data());
	assert(cudaGetLastError() == cudaSuccess);
}

void mapped_scan(int *const output, int *const input, const std::size_t size,
		 const int target_value) {
	const auto num_of_blocks = size / threads_per_block + 1;
	map_to_target<<<num_of_blocks, threads_per_block>>>(output, input, size,
							    target_value);
	assert(cudaGetLastError() == cudaSuccess);
	prefix_recursive(output, input, size);
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
	// using buffer_handle_t = GPUBufferHandle<tuple_t>;

	GPUStream cuda_stream;
	cudaError_t cuda_error; // Used to store and catch CUDA errors;
	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	PinnedCPUBuffer<tuple_t> cpu_tuple_buffer;

	// TODO: Do we use int or std::size_t for these buffers?
	PinnedCPUBuffer<int> cpu_hash_index;
	GPUBuffer<int> gpu_hash_index; // Stores modulo'd hash values for keys.
	GPUBuffer<int> scan;

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
			const auto handle = reinterpret_cast<GPUBuffer<tuple_t> *>(input);
#ifdef PARALLEL_PARTITION
			parallel_keyed_gpu_partition(*handle);
#else
			linear_keyed_gpu_partition(*handle);
#endif
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			const auto key = std::get<0>(t->getControlFields());
			const auto hashcode = hash(key);
			destination_index = routing_func(hashcode, num_of_destinations);
			this->ff_send_out_to(t, destination_index);
			return this->GO_ON;
		}
		delete input;
		return nullptr; // Silence potential compiler warnings.
	}

	/*
	 * Supposed to be slow, only for performance testing purposes.
	 */
	void linear_keyed_gpu_partition(const GPUBuffer<tuple_t> &handle) {
		const auto raw_batch_size = handle.size() * sizeof *handle.data();
		cpu_tuple_buffer.enlarge(handle.size());
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle.data(),
					     raw_batch_size, cudaMemcpyDeviceToHost,
					     cuda_stream.raw_stream());
		assert(cuda_error == cudaSuccess);
		std::vector<std::vector<tuple_t>> sub_buffers (num_of_destinations);

		cuda_stream.synchronize();
		for (auto i = 0; i < handle.size(); ++i) {
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
			GPUBuffer<tuple_t> sub_buffer {cpu_sub_buffer.size()};
			cuda_error = cudaMemcpyAsync(sub_buffer.data(), cpu_sub_buffer.data(),
						     raw_size, cudaMemcpyHostToDevice,
						     cuda_stream.raw_stream());
			assert(cuda_error == cudaSuccess);
			cuda_stream.synchronize();
			this->ff_send_out_to(new auto {std::move(sub_buffer)}, i);
		}
	}

	/*
	 * Actual partitioning implementation to be eventually used.
	 */
	void parallel_keyed_gpu_partition(const GPUBuffer<tuple_t> &handle) {
		cpu_tuple_buffer.enlarge(handle.size());
		cpu_hash_index.enlarge(handle.size());
		gpu_hash_index.enlarge(handle.size());
		scan.enlarge(handle.size());

		const auto raw_batch_size = handle.size() * sizeof *handle.data();
		// TODO: Can we avoid this copy on cpu?
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle.data(), raw_batch_size,
					     cudaMemcpyDeviceToHost, cuda_stream.raw_stream());
		assert(cuda_error == cudaSuccess);
		cuda_stream.synchronize();
		for (auto i = 0; i < handle->size(); ++i) {
			const auto key = std::get<0>(cpu_tuple_buffer[i].getControlFields());
			cpu_hash_index[i] = hash(key) % num_of_destinations;
		}
		const auto raw_index_size = sizeof *cpu_hash_index.data() * handle.size();
		cuda_error = cudaMemcpyAsync(gpu_hash_index.data(), cpu_hash_index.data(),
					     raw_index_size, cudaMemcpyHostToDevice,
					     cuda_stream.raw_stream());
		assert(cuda_error == cudaSuccess);

		const auto pow = get_closest_power_of_two(handle.size());
		for (auto i = 0; i < num_of_destinations; ++i) {
			mapped_scan(scan.data(), gpu_hash_index.data(), scan.size(), i);
			cuda_stream.synchronize();

			std::size_t bout_size;
			cuda_error = cudaMemcpyAsync(&bout_size, &scan[handle.size() - 1],
						     sizeof bout_size, cudaMemcpyDeviceToHost,
						     cuda_stream.raw_stream());
			assert(cuda_error == cudaSuccess);
			cuda_stream.synchronize();

			GPUBuffer<tuple_t> bout {bout_size};
			create_sub_batch<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream.raw_stream()>>>
				(handle.data(), num_of_destinations,
				 gpu_hash_index.data(), scan.data(), bout.data(), i);
			assert(cudaGetLastError() == cudaSuccess);
			cuda_stream.synchronize();
			ff_send_out_to(new auto {std::move(bout)}, i);
		}
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
