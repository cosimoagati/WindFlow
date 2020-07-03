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
/*
 * Parallel scan (prefix) implementation on GPU.  Code shamelessly taken from
 * the book "GPU Gems 3":
 * https://developer.nvidia.com/gpugems/gpugems3/contributors
 */
template <typename T>
__global__ void prescan(T *const g_odata, T *const g_idata, const int n) {
	extern __shared__ T temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];
	for (int d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 *thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) {
		temp[n - 1] = 0;  // clear the last element
	}
	for (int d = 1; d < n; d *= 2) {// traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

/*
 * Used to split a keyed batch to ensure keys always go to the same node.
 */
// TODO: Change names to something more self-explainatory.
template<typename tuple_t>
__global__ void create_sub_batch(tuple_t *const bin,
				 const std::size_t batch_size,
				 std::size_t *const index_gpu,
				 std::size_t *const scan_0,
				 const std::size_t bout_0,
				 const int target_node) {
	const auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_gpu[id] == target_node) {
		bout_0[scan_0[index[id]] - 1] = bin[id];
	}

}

template<typename tuple_t>
class Standard_EmitterGPU: public Basic_Emitter {
private:
	// type of the function to map the key hashcode onto an identifier
	// starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;

	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	std::size_t destination_index;
	std::size_t num_of_destinations;
	// TODO: is it fine for a GPU emitter to be used in a Tree_Emitter?
	// bool is_combined; // true if this node is used within a Tree_Emitter node
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
	{}

	Standard_EmitterGPU(const routing_func_t routing_func,
			    const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false)
		: routing_mode {FORWARD}, routing_func {routing_func},
		  destination_index {0},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{}

	// Why does this method exist? Isn't this redundant?
	Basic_Emitter *clone() const {
		return new Standard_EmitterGPU<tuple_t> {*this};
	}

	int svc_init() const { return 0; }

	void *svc(void *const input) {
		if (routing_mode != KEYBY) {
			return input; // Same thing whether input is a tuple or a GPU batch.
		}
		if (have_gpu_input) {
			auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
			std::size_t cpu_hash_index[num_of_destinations]; // Stores modulo"d hash values for keys.
			for (auto i = 0; i < handle->size; ++i) {
				cpu_hash_index[i]= hash(handle->buffer[i]);
			}
			const auto raw_index_size = sizeof(std::size_t) * num_of_destinations;
			std::size_t *gpu_hash_index;
			if (cudaMalloc(&gpu_hash_index, raw_index_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate GPU index");
			}
			cudaMemcpy(gpu_hash_index, cpu_hash_index, raw_index_size, cudaMemcpyHostToDevice);
			for (auto i = 0; i < num_of_destinations; ++i) {
				// TODO: Adapt parallel scan!
			}
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
		std::vector<tuple_t> cpu_tuple_buffer (handle->size);
		cudaMemcpy(cpu_tuple_buffer, handle->buffer,
			   sizeof(tuple_t) * handle->size, cudaMemcpyDeviceToHost);
		std::unordered_map<std::size_t, std::vector<tuple_t>> sub_buffer_map;

		for (auto i = 0; i < handle->size; ++i) {
			const auto &tuple = handle->buffer[i];
			auto hashcode = hash(tuple) % num_of_destinations;
			sub_buffer_map[hashcode].push_back(tuple);
		}
		for (const auto &kv : sub_buffer_map) {
			const auto &cpu_sub_buffer = kv->second;
			const auto raw_size = sizeof(tuple_t) * cpu_sub_buffer.size();
			tuple_t *gpu_sub_buffer;
			if (cudaMalloc(&gpu_sub_buffer, raw_size) != cudaSuccess) {
				failwith("Standard_EmitterGPU failed to allocate GPU sub-buffer");
			}
			cudaMemcpy(gpu_sub_buffer, cpu_sub_buffer. raw_size, cudaMemcpyHostToDevice);
			this->ff_send_out_to(new GPUBufferHandle {gpu_sub_buffer, cpu_sub_buffer.size()},
					     kv->first);
		}
		cudaFree(handle->buffer);
		delete handle;
	}

	void svc_end() const {}

	std::size_t getNDestinations() const { return num_of_destinations; }
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
