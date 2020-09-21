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
 *  @file    filter_gpu_node.hpp
 *  @author  Cosimo Agati
 *  @date    10/01/2019
 *
 *  @brief Individual FilterGPU node.
 *
 *  @section FilterGPU_Node (Description)
 *
 *  This file implements a single FilterGPU node, used by its internal farm.
 *  respect a given predicate given by the user.
 *
 *  The template parameter tuple_t must be default constructible, with a copy
 *  constructor and copy assignment operator, and it must provide and implement
 *  the setControlFields() and getControlFields() methods.
 */
#ifndef FILTER_GPU_NODE_H
#define FILTER_GPU_NODE_H

#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ff/node.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "gpu_utils.hpp"

namespace wf {
// TODO: Can we use a bit vector instead of a bool array?
template<typename tuple_t, typename func_t>
__global__ void run_filter_kernel(const func_t filter_func,
				  tuple_t *const tuple_buffer,
				  bool *const tuple_mask,
				  const std::size_t buffer_capacity) {
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride)
		filter_func(tuple_buffer[i], tuple_mask[i]);
}

template<typename tuple_t, typename func_t>
__global__ void run_filter_kernel_keyed(const func_t filter_func,
					tuple_t *const tuple_buffer,
					bool *const tuple_mask,
					TupleState *const tuple_state,
					const std::size_t scratchpad_size,
					const std::size_t buffer_capacity) {
	const auto num_threads = gridDim.x * blockDim.x;
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;

	for (auto i = 0u; i < buffer_capacity; ++i) {
		auto &state = tuple_state[i];
		if (state.hash % num_threads == index)
			filter_func(tuple_buffer[i], tuple_mask[i], state.scratchpad, scratchpad_size);
	}
}

template<typename tuple_t, typename func_t>
class FilterGPU_Node: public ff::ff_node_t<tuple_t> {
	template<typename F>
	static constexpr bool is_keyless = is_invocable<F, const tuple_t &, bool &>::value;
	template<typename F>
	static constexpr bool is_keyed = is_invocable<F, const tuple_t &, bool &, char *, std::size_t>::value;

	static_assert((is_keyless<func_t> && !is_keyed<func_t>) || (!is_keyless<func_t> && is_keyed<func_t>),
		      "WindFlow Error: filter function has an invalid signature: it is neither keyed nor keyless.");

	/*
	 * Class memebers
	 */
	func_t filter_func;
	std::string name;
	int total_buffer_capacity;
	int gpu_threads_per_block;
	int gpu_blocks;

	GPUStream cuda_stream;
	PinnedCPUBuffer<tuple_t> cpu_tuple_buffer;
	PinnedCPUBuffer<tuple_t> cpu_result_buffer;
	GPUBuffer<tuple_t> gpu_tuple_buffer;

	PinnedCPUBuffer<bool> cpu_tuple_mask;
	GPUBuffer<bool> gpu_tuple_mask;
	cudaError_t cuda_error; // Used to store and catch CUDA errors;

	int current_buffer_capacity {0};
	bool have_gpu_input;
	bool have_gpu_output;

	/*
	 *Only used for stateful (keyed) computations.
	 */
	std::unordered_map<key_t, GPUBuffer<char>> key_scratchpad_map;
	std::hash<key_t> hash;

	PinnedCPUBuffer<TupleState> cpu_tuple_state_buffer;
	GPUBuffer<TupleState> gpu_tuple_state_buffer;
	std::size_t scratchpad_size;
	bool was_batch_started {false};

	std::size_t buf_index {0};

#if defined(TRACE_WINDFLOW)
	unsigned long rcvTuples {0};
	double avg_td_us {0};
	double avg_ts_us {0};
	volatile unsigned long startTD, startTS, endTD, endTS;
	std::ofstream *logfile = nullptr;
#endif

	/*
	 * Helper function to ease transfer from host to device (and vice
	 * versa).  Assumes that both buffers share the same length, the common
	 * buffer capacity.
	 */
	template<typename T>
	void copy_host_buffer_to_device(T *device_to, T *host_from) {
		const auto size = total_buffer_capacity * sizeof(T);
		cuda_error = cudaMemcpyAsync(device_to, host_from, size,
					     cudaMemcpyHostToDevice, cuda_stream.raw());
		assert(cuda_error == cudaSuccess);
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	tuple_t *svc_aux(tuple_t *const t) {
		cpu_tuple_buffer[current_buffer_capacity] = *t;
		++current_buffer_capacity;
		delete t;
		if (current_buffer_capacity < total_buffer_capacity)
			return this->GO_ON;

		send_last_batch_if_any();
		copy_host_buffer_to_device(gpu_tuple_buffer.data(), cpu_tuple_buffer.data());
		for (auto i = 0; i < total_buffer_capacity; ++i)
			cpu_result_buffer[i] = cpu_tuple_buffer[i]; // Can we avoid these copies?

		current_buffer_capacity = 0;
		run_kernel();
		was_batch_started = true;
		return this->GO_ON;
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	tuple_t *svc_aux(tuple_t *const t) {
		cpu_tuple_buffer[current_buffer_capacity] = *t;
		const auto key = std::get<0>(t->getControlFields());
		allocate_scratchpad_if_not_present(key);
		cpu_tuple_state_buffer[current_buffer_capacity] = {hash(key), key_scratchpad_map[key].data()};
		delete t;

		++current_buffer_capacity;
		if (current_buffer_capacity < total_buffer_capacity)
			return this->GO_ON;

		send_last_batch_if_any();
		copy_host_buffer_to_device(gpu_tuple_buffer.data(), cpu_tuple_buffer.data());
		copy_host_buffer_to_device(gpu_tuple_state_buffer.data(), cpu_tuple_state_buffer.data());
		for (auto i = 0; i < total_buffer_capacity; ++i)
			cpu_result_buffer[i] = cpu_tuple_buffer[i]; // Can we avoid these copies?

		current_buffer_capacity = 0;
		run_kernel();
		was_batch_started = true;
		return this->GO_ON;
	}

	/*
	 * Allocates scratchpad on the device for the respective tuple, if not
	 * yet present, otherwise it does nothing.
	 */
	void allocate_scratchpad_if_not_present(const key_t &key) {
		if (key_scratchpad_map.find(key) != key_scratchpad_map.end())
			return;
		key_scratchpad_map[key] = scratchpad_size;
	}

	void send_last_batch_if_any() {
		if (!was_batch_started)
			return;

		cuda_stream.synchronize(); // is this sync redundant?
		const auto mask_size = total_buffer_capacity * sizeof(bool);
		cuda_error = cudaMemcpyAsync(cpu_tuple_mask.data(), gpu_tuple_mask.data(),
					     mask_size, cudaMemcpyDeviceToHost, cuda_stream.raw());
		assert(cuda_error == cudaSuccess);
		cuda_stream.synchronize();
		for (auto i = 0; i < total_buffer_capacity; ++i) {
			if (cpu_tuple_mask[i]) {
				this->ff_send_out(new tuple_t {cpu_result_buffer[i]});
				std::cout << "tuple sent!" << std::endl;
			}
		}
		std::cout << "batch sent!" << std::endl;
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	void run_kernel() {
		run_filter_kernel<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream.raw()>>>
			(filter_func, gpu_tuple_buffer.data(), gpu_tuple_mask.data(), total_buffer_capacity);
		assert(cudaGetLastError() == cudaSuccess);
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void run_kernel() {
		run_filter_kernel_keyed<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream.raw()>>>
			(filter_func, gpu_tuple_buffer.data(), gpu_tuple_mask.data(),
			 gpu_tuple_state_buffer.data(), scratchpad_size, total_buffer_capacity);
		assert(cudaGetLastError() == cudaSuccess);
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	void process_last_tuples() {
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			const auto &t = cpu_tuple_buffer[i];
			bool mask;

			filter_func(t, mask);
			if (mask)
				this->ff_send_out(new tuple_t {t});
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void process_last_tuples() {
		std::unordered_map<key_t, std::vector<char>> last_map;

		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto &t = cpu_tuple_buffer[i];
			const auto key = std::get<0>(t.getControlFields());

			if (last_map.find(key) == last_map.end()) {
				last_map.emplace(key, std::vector<char>(scratchpad_size));
				if (key_scratchpad_map.find(key) != key_scratchpad_map.end()) {
					cuda_error = cudaMemcpyAsync(last_map[key].data(),
								     key_scratchpad_map[key].data(),
								     scratchpad_size, cudaMemcpyDeviceToHost,
								     cuda_stream.raw());
					assert(cuda_error == cudaSuccess);
				}
			}
			bool mask;
			filter_func(t, mask, last_map[key].data(), scratchpad_size);
			if (mask)
				this->ff_send_out(new tuple_t {t});
		}
	}

public:
	FilterGPU_Node(const func_t filter_func, const std::string &name,
		       const int total_buffer_capacity,
		       const int gpu_threads_per_block,
		       const std::size_t scratchpad_size=0,
		       const bool have_gpu_input=false,
		       const bool have_gpu_output=false)
		: filter_func {filter_func}, name {name},
		  total_buffer_capacity {total_buffer_capacity},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(total_buffer_capacity
					/ static_cast<float>(gpu_threads_per_block))},
		  cpu_tuple_buffer {total_buffer_capacity},
		  cpu_result_buffer {total_buffer_capacity},
		  cpu_tuple_mask {total_buffer_capacity},
		  gpu_tuple_mask {total_buffer_capacity},
		  gpu_tuple_buffer {total_buffer_capacity},
		  scratchpad_size {scratchpad_size},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{
		if (is_keyed<func_t>) {
			cpu_tuple_state_buffer = total_buffer_capacity;
			gpu_tuple_state_buffer = total_buffer_capacity;
		}
	}

	// svc_init method (utilized by the FastFlow runtime)
	int svc_init() {
#if defined(TRACE_WINDFLOW)
		logfile = new std::ofstream();
		name += "_" + std::to_string(this->get_my_id()) + "_"
			+ std::to_string(getpid()) + ".log";
#if defined(LOG_DIR)
		std::string filename = std::string(STRINGIFY(LOG_DIR))
			+ "/" + name;
		std::string log_dir = std::string(STRINGIFY(LOG_DIR));
#else
		std::string filename = "log/" + name;
		std::string log_dir = std::string("log");
#endif
		// create the log directory
		if (mkdir(log_dir.c_str(), 0777) != 0) {
			struct stat st;
			if((stat(log_dir.c_str(), &st) != 0) || !S_ISDIR(st.st_mode)) {
				std::cerr << RED
					  << "WindFlow Error: directory for log files cannot be created"
					  << DEFAULT_COLOR << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}
		logfile->open(filename);

#endif
		return 0;
	}

	// svc method (utilized by the FastFlow runtime)
	tuple_t *svc(tuple_t *const t) {
#if defined(TRACE_WINDFLOW)
		startTS = current_time_nsecs();
		if (rcvTuples == 0)
			startTD = current_time_nsecs();
		rcvTuples++;
#endif
		const auto result = svc_aux(t);
#if defined(TRACE_WINDFLOW)
		endTS = current_time_nsecs();
		endTD = current_time_nsecs();
		double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
		avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
		double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
		avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
		startTD = current_time_nsecs();
#endif
		return result;
	}

	/*
	 * Acts on receiving the EOS (End Of Stream) signal by the FastFlow
	 * runtime.  It computes the last remaining tuples on the CPU, for
	 * simplicity, then sends out any remaining results from the last CUDA
	 * kernel.
	 */
	void eosnotify(ssize_t) {
		send_last_batch_if_any();
		process_last_tuples();
	}

	// svc_end method (utilized by the FastFlow runtime)
	void svc_end() {
#if defined(TRACE_WINDFLOW)
		std::ostringstream stream;
		stream << "************************************LOG************************************\n";
		stream << "No. of received tuples: " << rcvTuples << "\n";
		stream << "Average service time: " << avg_ts_us << " usec \n";
		stream << "Average inter-departure time: " << avg_td_us << " usec \n";
		stream << "***************************************************************************\n";
		*logfile << stream.str();
		logfile->close();
		delete logfile;
#endif
	}

	void set_GPUInput(const bool val) {
		have_gpu_input = val;
	}

	void set_GPUOutput(const bool val) {
		have_gpu_output = val;
	}

	/*
	 * This object may not be copied nor moved.
	 */
	FilterGPU_Node(const FilterGPU_Node &) = delete;
	FilterGPU_Node(FilterGPU_Node &&) = delete;
	FilterGPU_Node &operator=(const FilterGPU_Node &) = delete;
	FilterGPU_Node &operator=(FilterGPU_Node &&) = delete;
};

} // namespace wf

#endif
