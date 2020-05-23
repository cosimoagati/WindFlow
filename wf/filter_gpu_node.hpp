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

#include <ff/node.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp" // Probably not required...
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
	for (auto i = index; i < buffer_capacity; i += stride) {
		filter_func(tuple_buffer[i], tuple_mask[i]);
	}
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
		if (state.hash % num_threads == index) {
			filter_func(tuple_buffer[i], tuple_mask[i],
				    state.scratchpad, scratchpad_size);
		}
	}
}

template<typename tuple_t, typename func_t>
class FilterGPU_Node: public ff::ff_node_t<tuple_t> {
	template<typename F>
	static constexpr bool is_keyless = is_invocable<F, const tuple_t &,
							bool &>::value;
	template<typename F>
	static constexpr bool is_keyed = is_invocable<F, const tuple_t &,
						      bool &, char *,
						      std::size_t>::value;

	static_assert((is_keyless<func_t> && !is_keyed<func_t>)
		      || (!is_keyless<func_t> && is_keyed<func_t>),
		      "WindFlow Error: filter function has an invalid "
		      "signature: it is neither keyed nor keyless.");


	/*
	 * Class memebers
	 */
	func_t filter_func;
	std::string name;
	int total_buffer_capacity;
	int gpu_threads_per_block;
	int gpu_blocks;

	cudaStream_t cuda_stream;
	tuple_t *cpu_tuple_buffer;
	tuple_t *gpu_tuple_buffer;
	bool *cpu_tuple_mask;
	bool *gpu_tuple_mask;


	int current_buffer_capacity {0};


	/*
	 *Only used for stateful (keyed) computations.
	 */
	std::unordered_map<key_t, char *> key_scratchpad_map;
	std::hash<key_t> hash;
	TupleState *cpu_tuple_state_buffer;
	TupleState *gpu_tuple_state_buffer;
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
		cudaMemcpy(device_to, host_from, size, cudaMemcpyHostToDevice);
	}

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	void setup_tuple_state_buffers() {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void setup_tuple_state_buffers() {
		const auto size = total_buffer_capacity * sizeof(TupleState);
		if (cudaMallocHost(&cpu_tuple_state_buffer, size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate CPU tuple state buffer");
		}
		if (cudaMalloc(&gpu_tuple_state_buffer, size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate GPU tuple state buffer");
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	tuple_t *svc_aux(tuple_t *const t) {
		cpu_tuple_buffer[current_buffer_capacity] = *t;
		++current_buffer_capacity;
		delete t;
		if (current_buffer_capacity < total_buffer_capacity) {
			return this->GO_ON;
		}
		send_last_batch_if_any();
		copy_host_buffer_to_device(gpu_tuple_buffer, cpu_tuple_buffer);
		copy_host_buffer_to_device(gpu_tuple_mask, cpu_tuple_mask);
		current_buffer_capacity = 0;

		run_kernel();
		was_batch_started = true;
		return this->GO_ON;
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	tuple_t *svc_aux(tuple_t *const t) {
		cpu_tuple_buffer[current_buffer_capacity] = *t;
		const auto &key = std::get<0>(t->getControlFields());

		if (key_scratchpad_map.find(key) == key_scratchpad_map.end()) {
			auto &scratchpad = key_scratchpad_map[key];
			if (cudaMalloc(&scratchpad, scratchpad_size) != cudaSuccess) {
				failwith("FilterGPU_Node failed to allocate GPU scratchpad.");
			}
		}
		cpu_tuple_state_buffer[current_buffer_capacity] =
			{hash(key), key_scratchpad_map[key]};
		delete t;
		++current_buffer_capacity;

		if (current_buffer_capacity < total_buffer_capacity) {
			return this->GO_ON;
		}
		send_last_batch_if_any();
		copy_host_buffer_to_device(gpu_tuple_buffer, cpu_tuple_buffer);
		copy_host_buffer_to_device(gpu_tuple_state_buffer,
					   cpu_tuple_state_buffer);
		current_buffer_capacity = 0;
		run_kernel();
		was_batch_started = true;
		return this->GO_ON;
	}

	void send_last_batch_if_any() {
		if (!was_batch_started) {
			return;
		}
		cudaStreamSynchronize(cuda_stream);
		const auto size = total_buffer_capacity * sizeof(bool);
		cudaMemcpy(cpu_tuple_mask, gpu_tuple_mask, size,
			   cudaMemcpyDeviceToHost);
		for (auto i = 0; i < total_buffer_capacity; ++i) {
			if (cpu_tuple_mask[i]) {
				this->ff_send_out(new tuple_t {cpu_tuple_buffer[i]});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	void run_kernel() {
		run_filter_kernel<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(filter_func, gpu_tuple_buffer, gpu_tuple_mask,
			 total_buffer_capacity);
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void run_kernel() {
		run_filter_kernel_keyed<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(filter_func, gpu_tuple_buffer, gpu_tuple_mask,
			 gpu_tuple_state_buffer, scratchpad_size,
			 total_buffer_capacity);
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	void process_last_tuples() {
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			const auto &t = cpu_tuple_buffer[i];
			bool mask;

			filter_func(t, mask);
			if (mask) {
				this->ff_send_out(new tuple_t {t});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void process_last_tuples() {
		// This hashmap is used to copy the corresponding device
		// scratchpads to the host.  The map itself is used to make sure
		// we correctly preserve the scratchpad state for each key.
		std::unordered_map<key_t, char *> last_map;

		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto &t = cpu_tuple_buffer[i];
			const auto &key = std::get<0>(t.getControlFields());
			const auto &gpu_scratchpad = key_scratchpad_map[key];

			if (last_map.find(key) == last_map.end()) {
				last_map.emplace(key, new char[scratchpad_size]);
				cudaMemcpy(last_map[key].data(), gpu_scratchpad,
					   scratchpad_size, cudaMemcpyDeviceToHost);
			}
			bool mask;
			filter_func(t, mask, last_map[key].data(), scratchpad_size);
			if (mask) {
				this->ff_send_out(new tuple_t {t});
			}
		}
		for (auto &pair : last_map) {
			delete[] pair.second;
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_keyless<F>, int> = 0>
	void deallocate_tuple_state_buffers() {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void deallocate_tuple_state_buffers() {
		for (auto &pair : key_scratchpad_map) {
			cudaFree(pair.second);
		}
		cudaFreeHost(cpu_tuple_state_buffer);
		cudaFree(gpu_tuple_state_buffer);
	}

public:
	/*
	 * A single worker allocates both CPU and GPU buffers to store tuples.
	 * In case the function to be used is NOT in-place, it also allocates
	 * enough space for the CPU and GPU buffers to store the results.  The
	 * first constructor is used for keyless (stateless) version, the second is
	 * for the keyed (stateful) version.
	 */
	FilterGPU_Node(const func_t filter_func, const std::string &name,
		       const int total_buffer_capacity,
		       const int gpu_threads_per_block)
		: FilterGPU_Node {filter_func, name, total_buffer_capacity,
				  gpu_threads_per_block, 0}
	{}

	FilterGPU_Node(const func_t filter_func, const std::string &name,
		       const int total_buffer_capacity,
		       const int gpu_threads_per_block,
		       const std::size_t scratchpad_size)
		: filter_func {filter_func}, name {name},
		  total_buffer_capacity {total_buffer_capacity},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(total_buffer_capacity
					/ static_cast<float>(gpu_threads_per_block))},
		  scratchpad_size {scratchpad_size}
	{
		const auto tuple_buffer_size = sizeof(tuple_t) * total_buffer_capacity;
		if (cudaMallocHost(&cpu_tuple_buffer, tuple_buffer_size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate CPU tuple buffer");
		}
		const auto tuple_mask_size = sizeof(bool) * total_buffer_capacity;
		if (cudaMallocHost(&cpu_tuple_mask, tuple_mask_size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate CPU tuple mask");
		}
		if (cudaMalloc(&gpu_tuple_buffer, tuple_buffer_size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate GPU tuple buffer");
		}
		if (cudaMalloc(&gpu_tuple_mask, tuple_buffer_size) != cudaSuccess) {
			failwith("FilterGPU_Node failed to allocate GPU tuple mask");
		}
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in FilterGPU_Node");
		}
		setup_tuple_state_buffers();
	}

	~FilterGPU_Node() {
		cudaFreeHost(cpu_tuple_buffer);
		cudaFreeHost(cpu_tuple_mask);
		cudaFree(gpu_tuple_mask);
		deallocate_tuple_state_buffers();
		cudaStreamDestroy(cuda_stream);

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
		cudaFree(cpu_tuple_buffer);
		cudaFree(cpu_tuple_mask);
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
