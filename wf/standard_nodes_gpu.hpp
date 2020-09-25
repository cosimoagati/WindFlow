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

# define PARALLEL_PARTITION

namespace wf {
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
	if (id < batch_size && index[id] == target_node)
		bout[scan[index[id]] - 1] = bin[id];
}

template<typename tuple_t>
class Standard_EmitterGPU: public Basic_Emitter {
private:
	// type of the function to map the key hashcode onto an identifier
	// starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;
	// using buffer_handle_t = GPUBufferHandle<tuple_t>;

	GPUStream stream;
	cudaError_t cuda_error; // Used to store and catch CUDA errors;
	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	PinnedCPUBuffer<tuple_t> cpu_tuple_buffer;

	// TODO: Do we use int or std::size_t for these buffers?
	PinnedCPUBuffer<std::size_t> cpu_hash_index;
	GPUBuffer<std::size_t> gpu_hash_index; // Stores modulo'd hash values for keys.
	GPUBuffer<std::size_t> scan;

	std::size_t destination_index {0};
	std::size_t num_of_destinations;
	int gpu_blocks;
	int gpu_threads_per_block;

	// TODO: is it fine for a GPU emitter to be used in a Tree_Emitter?
	bool is_combined; // true if this node is used within a Tree_Emitter node
	bool have_gpu_input;

public:
	Standard_EmitterGPU(const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const int gpu_threads_per_block=256)
		: routing_mode {FORWARD},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(num_of_destinations
					/ static_cast<float>(gpu_threads_per_block))}
	{
		if (num_of_destinations <= 0)
			failwith("Standard_EmitterGPU initialized with non-positive number of destinations.");
	}

	Standard_EmitterGPU(const routing_func_t routing_func,
			    const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const int gpu_threads_per_block=256)
		: routing_mode {KEYBY}, routing_func {routing_func},
		  num_of_destinations {num_of_destinations},
		  cpu_tuple_buffer (num_of_destinations),
		  cpu_hash_index (num_of_destinations),
		  gpu_hash_index {num_of_destinations},
		  scan {num_of_destinations}, have_gpu_input {have_gpu_input},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(num_of_destinations
					/ static_cast<float>(gpu_threads_per_block))}
	{
		if (num_of_destinations <= 0)
			failwith("Standard_EmitterGPU initialized with non-positive number of destinations.");
	}

	Basic_Emitter *clone() const {
		return new Standard_EmitterGPU<tuple_t> {*this};
	}

	int svc_init() const { return 0; }

	void *svc(void *const input) {
		if (routing_mode != KEYBY)
			return input; // Same whether input is a tuple or batch.
		if (have_gpu_input) {
			const auto handle = reinterpret_cast<GPUBuffer<tuple_t> *>(input);
#ifdef PARALLEL_PARTITION
			parallel_keyed_gpu_partition(*handle);
#else
			linear_keyed_gpu_partition(*handle);
#endif
			delete handle;
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			const auto key = std::get<0>(t->getControlFields());
			const auto hashcode = hash(key);
			destination_index = routing_func(hashcode, num_of_destinations);
			this->ff_send_out_to(t, destination_index);
		}
		return this->GO_ON;
	}

	/*
	 * Supposed to be slow, only for performance testing purposes.
	 */
	void linear_keyed_gpu_partition(const GPUBuffer<tuple_t> &handle) {
		const auto raw_batch_size = handle.size() * sizeof *handle.data();
		cpu_tuple_buffer.resize(handle.size());
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle.data(),
					     raw_batch_size, cudaMemcpyDeviceToHost,
					     stream.raw());
		assert(cuda_error == cudaSuccess);
		std::vector<std::vector<tuple_t>> sub_buffers (num_of_destinations);

		stream.sync();
		for (auto i = 0; i < handle.size(); ++i) {
			const auto &tuple = cpu_tuple_buffer[i];
			const auto key = std::get<0>(tuple.getControlFields());
			const auto hashcode = hash(key) % num_of_destinations;
			sub_buffers[hashcode].push_back(tuple);
		}
		for (auto i = 0; i < num_of_destinations; ++i) {
			if (sub_buffers[i].empty())
				continue;

			const auto &cpu_sub_buffer = sub_buffers[i];
			const auto raw_size = sizeof *cpu_sub_buffer.data() * cpu_sub_buffer.size();
			GPUBuffer<tuple_t> sub_buffer {cpu_sub_buffer.size()};
			cuda_error = cudaMemcpyAsync(sub_buffer.data(), cpu_sub_buffer.data(),
						     raw_size, cudaMemcpyHostToDevice, stream.raw());
			assert(cuda_error == cudaSuccess);
			stream.sync();
			this->ff_send_out_to(new auto {std::move(sub_buffer)}, i);
		}
	}

	/*
	 * Actual partitioning implementation to be eventually used.
	 */
	void parallel_keyed_gpu_partition(const GPUBuffer<tuple_t> &handle) {
		cpu_tuple_buffer.resize(handle.size());
		cpu_hash_index.resize(handle.size());
		gpu_hash_index.resize(handle.size());
		scan.resize(handle.size());

		const auto raw_batch_size = handle.size() * sizeof *handle.data();
		// TODO: Can we avoid this copy on cpu?
		cuda_error = cudaMemcpyAsync(cpu_tuple_buffer.data(), handle.data(), raw_batch_size,
					     cudaMemcpyDeviceToHost, stream.raw());
		assert(cuda_error == cudaSuccess);
		stream.sync();

		for (auto i = 0; i < handle.size(); ++i) {
			const auto key = std::get<0>(cpu_tuple_buffer[i].getControlFields());
			cpu_hash_index[i] = hash(key) % num_of_destinations;
		}
		const auto raw_index_size = sizeof *cpu_hash_index.data() * handle.size();
		cuda_error = cudaMemcpyAsync(gpu_hash_index.data(), cpu_hash_index.data(),
					     raw_index_size, cudaMemcpyHostToDevice, stream.raw());
		assert(cuda_error == cudaSuccess);

		for (auto i = 0; i < num_of_destinations; ++i) {
			mapped_scan(scan, gpu_hash_index, scan.size(), static_cast<std::size_t>(i), stream);
			stream.sync();

			std::size_t bout_size;
			cuda_error = cudaMemcpyAsync(&bout_size, scan.data() + handle.size() - 1,
						     sizeof bout_size, cudaMemcpyDeviceToHost,
						     stream.raw());
			assert(cuda_error == cudaSuccess);
			stream.sync();

			GPUBuffer<tuple_t> bout {bout_size};
			create_sub_batch<<<gpu_blocks, gpu_threads_per_block, 0, stream.raw()>>>
				(handle.data(), handle.size(), gpu_hash_index.data(),
				 scan.data(), bout.data(), i);
			assert(cudaGetLastError() == cudaSuccess);
			stream.sync();
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
