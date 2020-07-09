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
 *  @file    map_gpu.hpp
 *  @author  Cosimo Agati
 *  @date    08/01/2019
 *
 *  @brief MapGPU nodes, to be used by the MapGPU operators as workers.
 *
 *  @section MapGPU_Node (Description)
 *
 *  The template parameters tuple_t and result_t must be default constructible,
 *  with a copy Constructor and copy assignment operator, and they must provide
 *  and implement the setControlFields() and getControlFields() methods.
 */

#ifndef MAP_GPU_NODE_H
#define MAP_GPU_NODE_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ff/node.hpp>
#include "basic.hpp"
#include "gpu_utils.hpp"

namespace wf {
// N.B.: CUDA __global__ kernels must not be member functions.
/**
 * \brief Map kernel (in-place keyless version). Run function and store results in same buffer.
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer pointer to the start of the buffered tuples.
 * \param buffer_capacity How many tuples the buffer contains.
 */
template<typename tuple_t, typename func_t>
__global__ void run_map_kernel_ip(const func_t map_func,
				  tuple_t *const tuple_buffer,
				  const std::size_t buffer_capacity) {
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i]);
	}
}

/**
 * \brief Map kernel (non-in-place keyless version). Run function and store results a new buffer.
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer pointer to the start of the buffered tuples
 * \param result_buffer pointer to the start of the buffer that will contain the results
 * \param buffer_capacity How many tuples the buffer contains.
 */
template<typename tuple_t, typename result_t, typename func_t>
__global__ void run_map_kernel_nip(const func_t map_func,
				   tuple_t *const tuple_buffer,
				   result_t *const result_buffer,
				   const std::size_t buffer_capacity) {
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i], result_buffer[i]);
	}
}

/**
 * \brief Map kernel (in-place keyed version). Run function and store results a new buffer.
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer Pointer to the start of the buffered tuples.
 * \param tuple_state Pointer to the start of a tuple state array.
 * \param scratchpad_size Size of an individual scratchpad.
 * \param buffer_capacity How many tuples the buffer contains.
 */
template<typename tuple_t, typename func_t>
__global__ void run_map_kernel_keyed_ip(const func_t map_func,
					tuple_t *const tuple_buffer,
					TupleState *const tuple_state,
					const std::size_t scratchpad_size,
					const std::size_t buffer_capacity) {
	const auto num_threads = gridDim.x * blockDim.x;
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;

	for (auto i = 0u; i < buffer_capacity; ++i) {
		auto &state = tuple_state[i];
		if (state.hash % num_threads == index) {
			map_func(tuple_buffer[i], state.scratchpad, scratchpad_size);
		}
	}
}

/**
 * \brief Map kernel (non in-place keyed version). Run function and store results a new buffer
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer pointer to the start of the buffered tuples
 * \param tuple_state pointer to the start of a tuple state array.
 * \param result_buffer pointer to the start of the buffer that will contain the results.
 * \param buffer_capacity How many tuples the buffer contains.
 * \param scratchpad_size Size of an individual scratchpad.
 */
template<typename tuple_t, typename result_t, typename func_t>
__global__ void run_map_kernel_keyed_nip(const func_t map_func,
					 tuple_t *const tuple_buffer,
					 TupleState *const tuple_state,
					 result_t *const result_buffer,
					 const std::size_t scratchpad_size,
					 const std::size_t buffer_capacity) {
	const auto num_threads = gridDim.x * blockDim.x;
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;

	for (auto i = 0u; i < buffer_capacity; ++i) {
		auto &state = tuple_state[i];
		if (state.hash % num_threads == index) {
			map_func(tuple_buffer[i], result_buffer[i],
				 state.scratchpad, scratchpad_size);
		}
	}
}

template<typename tuple_t, typename result_t, typename func_t>
class MapGPU_Node: public ff::ff_minode {
	/*
	 * Name function properties, used to verify compile-time invariants and
	 * only compile the required member functions.  These predicates cannot
	 * be declared as auto, since they would be treated as redeclarations
	 * otherwise (without an explicit type, they would be considered
	 * incompatible with any previous instantiation, that's why using them
	 * just once with auto works, but not if used more than once).
	 */
	template<typename F>
	static constexpr bool is_in_place_keyless = is_invocable<F, tuple_t &>::value;

	template<typename F>
	static constexpr bool is_not_in_place_keyless = is_invocable<F, const tuple_t &, tuple_t &>::value;

	template<typename F>
	static constexpr bool is_in_place_keyed = is_invocable<F, tuple_t &, char *, std::size_t>::value;

	template<typename F>
	static constexpr bool is_not_in_place_keyed = is_invocable<F, const tuple_t &,
								   result_t &, char *, std::size_t>::value;

	template<typename F>
	static constexpr bool is_in_place = is_in_place_keyless<F> || is_in_place_keyed<F>;

	template<typename F>
	static constexpr bool is_keyed = is_in_place_keyed<F> || is_not_in_place_keyed<F>;

	static_assert((!is_in_place<func_t> == (is_not_in_place_keyless<func_t>
						|| is_not_in_place_keyed<func_t>))
		      && (!is_keyed<func_t> == (is_in_place_keyless<func_t>
						|| is_not_in_place_keyless<func_t>)),
		      "Error: Negating predicates does not work as expected.");

	static_assert((is_in_place_keyless<func_t> && !is_not_in_place_keyless<func_t>
		       && !is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
		      || (!is_in_place_keyless<func_t> && is_not_in_place_keyless<func_t>
			  && !is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
		      || (!is_in_place_keyless<func_t> && !is_not_in_place_keyless<func_t>
			  && is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
		      || (!is_in_place_keyless<func_t> && !is_not_in_place_keyless<func_t>
			  && !is_in_place_keyed<func_t> && is_not_in_place_keyed<func_t>),
		      "WindFlow Error: MapGPU function parameter does not have "
		      "a valid signature. It must be EXACTLY one of:\n"
		      "void(tuple_t &) (In-place, keyless)\n"
		      "void(const tuple_t, result_t &) (Non in-place, keyless)\n"
		      "void(tuple_t &, char *, std::size_t) (In-place, keyed)\n"
		      "void(const tuple_t &, result_t &, char *, std::size_t) (Non in-place, keyed)");
	/*
	 * If the function to be computed is in-place, check if the input
	 * type (tuple_t) is the same as the output type (result_t).  This
	 * greatly simplifies class implementation, since we can now send out
	 * objects of type tuple_t without having to use casts.
	 *
	 * How does this work? Remember that A -> B is equivalent to !A || B in
	 * Boolean logic!
	 */
	static_assert(!is_in_place<func_t> || std::is_same<tuple_t, result_t>::value,
		      "WindFlow Error: if instantiating MapGPU with an in-place "
		      "function, the input type and the output type must match.");

	/*
	 * Type of the tuple key for keyed computations, obtained by getting the
	 * respective value from a temporary tuple.
	 */
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;

	/*
	 * Class memebers.
	 */
	func_t map_func;
	std::string operator_name;
	int total_buffer_capacity;
	int gpu_threads_per_block;
	int gpu_blocks;
	bool have_gpu_input;
	bool have_gpu_output;

	int current_buffer_capacity {0};

	cudaStream_t cuda_stream;
	tuple_t *cpu_tuple_buffer;
	tuple_t *gpu_tuple_buffer;
	result_t *cpu_result_buffer;
	result_t *gpu_result_buffer;

	/*
	 * Only used for stateful (keyed) computations.
	 */
	std::unordered_map<key_t, char *> key_scratchpad_map;
	std::hash<key_t> hash;
	TupleState *cpu_tuple_state_buffer;
	TupleState *gpu_tuple_state_buffer;
	std::size_t scratchpad_size;

	bool was_batch_started {false};

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

	template<typename F=func_t, typename std::enable_if_t<is_in_place<F>, int> = 0>
	void allocate_gpu_tuple_input_buffer() {}

	template<typename F=func_t, typename std::enable_if_t<!is_in_place<F>, int> = 0>
	void allocate_gpu_tuple_input_buffer() {
		const auto tuple_buffer_size = sizeof(tuple_t) * total_buffer_capacity;
		if (cudaMalloc(&gpu_tuple_buffer, tuple_buffer_size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU tuple buffer");
		}
	}

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	void setup_tuple_state_buffers() {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void setup_tuple_state_buffers() {
		const auto size = total_buffer_capacity * sizeof(TupleState);
		if (cudaMallocHost(&cpu_tuple_state_buffer, size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate CPU tuple state buffer");
		}
		if (cudaMalloc(&gpu_tuple_state_buffer, size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU tuple state buffer");
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyless<F>, int> = 0>
	void *svc_aux(void *const input) {
		if (have_gpu_input) {
			const auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<tuple_t> {gpu_result_buffer,
											total_buffer_capacity});
				} else {
					send_tuples_to_cpu_operator();
					cudaFree(gpu_result_buffer);
					enlarge_cpu_buffer(cpu_result_buffer, handle->size);
				}
			}
			gpu_result_buffer = handle->buffer;
			total_buffer_capacity = handle->size;
			delete handle;
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			cpu_tuple_buffer[current_buffer_capacity] = *t;
			delete t;
			++current_buffer_capacity;
			if (current_buffer_capacity < total_buffer_capacity) {
				return this->GO_ON;
			}
			assert(current_buffer_capacity == total_buffer_capacity);
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<tuple_t> {gpu_result_buffer,
											total_buffer_capacity});
					const auto size = total_buffer_capacity * sizeof(tuple_t);
					while (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
						// Empty loop body.
					}
				} else {
					send_tuples_to_cpu_operator();
				}
			}
			copy_host_buffer_to_device(gpu_result_buffer, cpu_tuple_buffer);
			current_buffer_capacity = 0;
		}
		run_map_kernel_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_result_buffer, total_buffer_capacity);
		was_batch_started = true;
		return this->GO_ON;
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyless<F>, int> = 0>
	void *svc_aux(void *const input) {
		if (have_gpu_input) {
			const auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
											 total_buffer_capacity});
				} else {
					send_tuples_to_cpu_operator();
					cudaFree(gpu_result_buffer);
					enlarge_cpu_buffer(cpu_result_buffer, handle->size);
				}
			}
			gpu_result_buffer = reinterpret_cast<result_t *>(gpu_tuple_buffer);
			enlarge_gpu_buffer(gpu_result_buffer, handle->size);
			gpu_tuple_buffer = handle->buffer;
			total_buffer_capacity = handle->size;
			delete handle;
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			cpu_tuple_buffer[current_buffer_capacity] = *t;
			++current_buffer_capacity;
			delete t;
			if (current_buffer_capacity < total_buffer_capacity) {
				return this->GO_ON;
			}
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
											 total_buffer_capacity});
					const auto size = total_buffer_capacity * sizeof(result_t);
					while (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
						// Empty loop body.
					}
				} else {
					send_tuples_to_cpu_operator();
				}
			}
			// copy_host_buffer_to_device(gpu_tuple_buffer, cpu_tuple_buffer);
			cudaMemcpy(gpu_tuple_buffer, cpu_tuple_buffer,
				   total_buffer_capacity * sizeof(tuple_t), cudaMemcpyHostToDevice);
			current_buffer_capacity = 0;
		}
		run_map_kernel_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 total_buffer_capacity);
		was_batch_started = true;
		return this->GO_ON;
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyed<F>, int> = 0>
	void *svc_aux(void *const input) {
		if (have_gpu_input) {
			const auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<tuple_t> {gpu_result_buffer,
											total_buffer_capacity});
				} else {
					send_tuples_to_cpu_operator();
					cudaFree(gpu_result_buffer);
					enlarge_cpu_buffer(cpu_result_buffer, handle->size);
				}
			}
			gpu_result_buffer = handle->buffer;
			total_buffer_capacity = handle->size;
			delete handle;
			enlarge_cpu_buffer(cpu_tuple_state_buffer, total_buffer_capacity);

			// Ensure scratchpads are allocated and retrieve keys. Could be costly...
			for (auto i = 0; i < total_buffer_capacity; ++i) {
				tuple_t dummy_tuple;
				cudaMemcpy(&dummy_tuple, &gpu_result_buffer[i],
					   sizeof(tuple_t), cudaMemcpyDeviceToHost);
				const auto key = std::get<0>(dummy_tuple.getControlFields());
				allocate_scratchpad_if_not_present(key);
				cpu_tuple_state_buffer[i] = {hash(key), key_scratchpad_map[key]};
			}
			copy_host_buffer_to_device(gpu_tuple_state_buffer, cpu_tuple_state_buffer);
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			cpu_tuple_buffer[current_buffer_capacity] = *t;
			const auto key = std::get<0>(t->getControlFields());
			delete t;
			allocate_scratchpad_if_not_present(key);

			cpu_tuple_state_buffer[current_buffer_capacity] =
				{hash(key), key_scratchpad_map[key]};
			++current_buffer_capacity;
			if (current_buffer_capacity < total_buffer_capacity) {
				return this->GO_ON;
			}
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<tuple_t> {gpu_result_buffer,
											total_buffer_capacity});
					const auto size = total_buffer_capacity * sizeof(tuple_t);
					while (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
						// Empty loop body.
					}
				} else {
					send_tuples_to_cpu_operator();
				}
			}
			copy_host_buffer_to_device(gpu_result_buffer, cpu_tuple_buffer);
			copy_host_buffer_to_device(gpu_tuple_state_buffer,
						   cpu_tuple_state_buffer);
			current_buffer_capacity = 0;
		}
		run_map_kernel_keyed_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_result_buffer, gpu_tuple_state_buffer,
			 scratchpad_size, total_buffer_capacity);
		was_batch_started = true;
		return this->GO_ON;
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyed<F>, int> = 0>
	void *svc_aux(void *const input) {
		if (have_gpu_input) {
			cudaFree(gpu_tuple_buffer);
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
											 total_buffer_capacity});
					const auto size = total_buffer_capacity * sizeof(result_t);
					while (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
						// Empty loop body.
					}
				} else {
					send_tuples_to_cpu_operator();
				}
			}
			const auto handle = reinterpret_cast<GPUBufferHandle<tuple_t> *>(input);
			gpu_tuple_buffer = handle->buffer;
			total_buffer_capacity = handle->size;
			delete handle;
			enlarge_gpu_buffer(gpu_result_buffer);

			// Ensure scratchpads are allocated. Could be costly...
			for (auto i = 0; i < total_buffer_capacity; ++i) {
				tuple_t dummy_tuple;
				cudaMemcpy(&dummy_tuple, &gpu_tuple_buffer[i],
					   sizeof(tuple_t), cudaMemcpyDeviceToHost);
				const auto key = std::get<0>(dummy_tuple.getControlFields());
				allocate_scratchpad_if_not_present(key);
				cpu_tuple_state_buffer[i] = {hash(key), key_scratchpad_map[key]};
			}
			copy_host_buffer_to_device(gpu_tuple_state_buffer, cpu_tuple_state_buffer);
		} else {
			const auto t = reinterpret_cast<tuple_t *>(input);
			cpu_tuple_buffer[current_buffer_capacity] = *t;
			const auto key = std::get<0>(t->getControlFields());
			delete t;
			allocate_scratchpad_if_not_present(key);

			cpu_tuple_state_buffer[current_buffer_capacity] =
				{hash(key), key_scratchpad_map[key]};
			++current_buffer_capacity;
			if (current_buffer_capacity < total_buffer_capacity) {
				return this->GO_ON;
			}
			if (was_batch_started) {
				cudaStreamSynchronize(cuda_stream);
				if (have_gpu_output) {
					this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
											 total_buffer_capacity});
					const auto size = total_buffer_capacity * sizeof(result_t);
					while (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
						// Empty loop body.
					}
				} else {
					send_tuples_to_cpu_operator();
				}
			}
			copy_host_buffer_to_device(gpu_tuple_buffer, cpu_tuple_buffer);
			copy_host_buffer_to_device(gpu_tuple_state_buffer,
						   cpu_tuple_state_buffer);
			current_buffer_capacity = 0;
		}
		run_map_kernel_keyed_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 gpu_tuple_state_buffer, scratchpad_size,
			 total_buffer_capacity);
		was_batch_started = true;
		return this->GO_ON;
	}

	void send_tuples_to_cpu_operator() {
		const auto size = total_buffer_capacity * sizeof(result_t);
		cudaMemcpy(cpu_result_buffer, gpu_result_buffer, size,
			   cudaMemcpyDeviceToHost);
		for (auto i = 0; i < total_buffer_capacity; ++i) {
			this->ff_send_out(new result_t {cpu_result_buffer[i]});
		}
	}

	/*
	 * Resizes buffer to new_size if smaller. the buffer must be allocated
	 * on the host via CUDA.
	 */
	template<typename T>
	void enlarge_cpu_buffer(T *&buffer, const int new_size) {
		if (total_buffer_capacity >= new_size) {
			return;
		}
		cudaFreeHost(buffer);
		const auto buffer_size = new_size * sizeof(T);
		while (cudaMallocHost(&buffer, buffer_size) != cudaSuccess) {
			// Empty loop body.
		}

	}

	/*
	 * Resizes buffer to new_size if smaller. the buffer must be allocated
	 * on the device via CUDA.
	 */
	template<typename T>
	void enlarge_gpu_buffer(T *&buffer, const int new_size) {
		if (total_buffer_capacity >= new_size) {
			return;
		}
		cudaFree(buffer);
		const auto buffer_size = new_size * sizeof(T);
		while (cudaMalloc(&buffer, buffer_size) != cudaSuccess) {
			// Empty loop body.
		}
	}

	/*
	 * Allocates scratchpad on the device for the respective tuple, if not
	 * yet present, otherwise it does nothing.
	 */
	void allocate_scratchpad_if_not_present(const key_t &key) {
		if (key_scratchpad_map.find(key) != key_scratchpad_map.end()) {
			return;
		}
		if (cudaMalloc(&key_scratchpad_map[key], scratchpad_size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU scratchpad.");
		}
	}

	/*
	 * In case of input from a host (CPU) operator, we must process any
	 * remaining tuples in the buffer at the end of the stream.
	 */
	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyless<F>, int> = 0>
	void process_last_buffered_tuples() {
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			map_func(cpu_tuple_buffer[i]);
		}
		if (have_gpu_output) {
			const auto size = current_buffer_capacity * sizeof(result_t);
			cudaMemcpy(gpu_result_buffer, cpu_tuple_buffer, size,
				   cudaMemcpyHostToDevice);
			this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
									 current_buffer_capacity});
		} else {
			for (auto i = 0; i < current_buffer_capacity; ++i) {
				this->ff_send_out(new result_t {cpu_tuple_buffer[i]});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyless<F>, int> = 0>
	void process_last_buffered_tuples() {
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			map_func(cpu_tuple_buffer[i], cpu_result_buffer[i]);
		}
		if (have_gpu_output) {
			const auto size = current_buffer_capacity * sizeof(result_t);
			cudaMemcpy(gpu_result_buffer, cpu_result_buffer, size,
				   cudaMemcpyHostToDevice);
			this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
									 current_buffer_capacity});
		} else {
			for (auto i = 0; i < current_buffer_capacity; ++i) {
				this->ff_send_out(new result_t {cpu_result_buffer[i]});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyed<F>, int> = 0>
	void process_last_buffered_tuples() {
		std::unordered_map<key_t, std::vector<char>> last_map;

		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto &t = cpu_tuple_buffer[i];
			const auto key = std::get<0>(t.getControlFields());

			if (last_map.find(key) == last_map.end()) {
				last_map.emplace(key, std::vector<char>(scratchpad_size));
				if (key_scratchpad_map.find(key) != key_scratchpad_map.end()) {
					cudaMemcpy(last_map[key].data(), key_scratchpad_map[key],
						   scratchpad_size, cudaMemcpyDeviceToHost);
				}
			}
			map_func(t, last_map[key].data(), scratchpad_size);
		}
		if (have_gpu_output) {
			const auto size = current_buffer_capacity * sizeof(result_t);
			cudaMemcpy(gpu_result_buffer, cpu_tuple_buffer,
				   size, cudaMemcpyHostToDevice);
			this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
									 current_buffer_capacity});
		} else {
			for (auto i = 0; i < current_buffer_capacity; ++i) {
				this->ff_send_out(new result_t {cpu_tuple_buffer[i]});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyed<F>, int> = 0>
	void process_last_buffered_tuples() {
		std::unordered_map<key_t, std::vector<char>> last_map;

		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto &t = cpu_tuple_buffer[i];
			const auto key = std::get<0>(t.getControlFields());

			if (last_map.find(key) == last_map.end()) {
				last_map.emplace(key, std::vector<char>(scratchpad_size));
				if (key_scratchpad_map.find(key) != key_scratchpad_map.end()) {
					cudaMemcpy(last_map[key].data(), key_scratchpad_map[key],
						   scratchpad_size, cudaMemcpyDeviceToHost);
				}
			}
			map_func(t, cpu_result_buffer[i],
				 last_map[key].data(), scratchpad_size);
		}
		if (have_gpu_output) {
			const auto size = current_buffer_capacity * sizeof(result_t);
			cudaMemcpy(gpu_result_buffer, cpu_result_buffer, size,
				   cudaMemcpyHostToDevice);
			this->ff_send_out(new GPUBufferHandle<result_t> {gpu_result_buffer,
									 current_buffer_capacity});
		} else {
			for (auto i = 0; i < current_buffer_capacity; ++i) {
				this->ff_send_out(new result_t {cpu_result_buffer[i]});
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place<F>, int> = 0>
	void deallocate_gpu_tuple_input_buffer() const {}

	template<typename F=func_t, typename std::enable_if_t<!is_in_place<F>, int> = 0>
	void deallocate_gpu_tuple_input_buffer() { cudaFree(gpu_tuple_buffer); }

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	void deallocate_tuple_state_buffers() const {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void deallocate_tuple_state_buffers() {
		for (auto &pair : key_scratchpad_map) {
			cudaFree(pair.second);
		}
		cudaFreeHost(cpu_tuple_state_buffer);
		cudaFree(gpu_tuple_state_buffer);
	}
public:
	MapGPU_Node(const func_t map_func, const std::string &name,
		    const int total_buffer_capacity,
		    const int gpu_threads_per_block,
		    const std::size_t scratchpad_size=0,
		    const bool have_gpu_input=false,
		    const bool have_gpu_output=false)
		: map_func {map_func}, operator_name {name},
		  total_buffer_capacity {total_buffer_capacity},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(total_buffer_capacity
					/ static_cast<float>(gpu_threads_per_block))},
		  scratchpad_size {scratchpad_size},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{
		assert(total_buffer_capacity > 0 && gpu_threads_per_block > 0
		       && scratchpad_size >= 0);
		const auto tuple_buffer_size = sizeof(tuple_t) * total_buffer_capacity;
		if (!have_gpu_input) {
			if (cudaMallocHost(&cpu_tuple_buffer, tuple_buffer_size) != cudaSuccess) {
				failwith("MapGPU_Node failed to allocate CPU tuple buffer");
			}
		}
		const auto result_buffer_size = sizeof(result_t) * total_buffer_capacity;
		if (!have_gpu_input) {
			if (cudaMalloc(&gpu_result_buffer, result_buffer_size) != cudaSuccess) {
				failwith("MapGPU_Node failed to allocate GPU result buffer");
			}
		}
		if (!have_gpu_output) {
			if (cudaMallocHost(&cpu_result_buffer, result_buffer_size) != cudaSuccess) {
				failwith("MapGPU_Node failed to allocate CPU result buffer");
			}
		}
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		}
		allocate_gpu_tuple_input_buffer();
		setup_tuple_state_buffers();
	}

	~MapGPU_Node() {
		cudaFreeHost(cpu_tuple_buffer);
		cudaFreeHost(cpu_result_buffer);
		cudaFree(gpu_result_buffer);
		deallocate_gpu_tuple_input_buffer();
		deallocate_tuple_state_buffers();
		cudaStreamDestroy(cuda_stream);
	}

	// svc_init method (utilized by the FastFlow runtime)
	int svc_init() override {
#if defined(TRACE_WINDFLOW)
		logfile = new std::ofstream();
		operator_name += "_node_" + std::to_string(get_my_id())
			+ ".log";
		const auto filename =
			std::string {STRINGIFY(TRACE_WINDFLOW)}+ "/" + operator_name;

		logfile->open(filename);
		if (logfile->fail()) {
			failwith("Error opening MapGPU_Node log file.");
		}
#endif
		return 0;
	}

	/*
	 * svc function used by the FastFlow runtime.  It calls the appropriate
	 * auxiliary function based on whether the function is stateless or not
	 * (keyed).
	 */
	void *svc(void *const t) override {
#if defined (TRACE_WINDFLOW)
		startTS = current_time_nsecs();
		if (rcvTuples == 0) {
			startTD = current_time_nsecs();
		}
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
	void eosnotify(ssize_t) override  {
		if (was_batch_started) {
			cudaStreamSynchronize(cuda_stream);
			if (have_gpu_output) {
				this->ff_send_out(new GPUBufferHandle<result_t>{gpu_result_buffer,
										total_buffer_capacity});
			} else {
				send_tuples_to_cpu_operator();
			}
		}
		if (!have_gpu_input) {
			process_last_buffered_tuples();
		}
	}

	// svc_end method (utilized by the FastFlow runtime)
	void svc_end() override {
#if defined (TRACE_WINDFLOW)
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

	void set_GPUInput(const bool new_val) {
		if (have_gpu_input == new_val) {
			return;
		}
		// TODO: Reallocate buffers...
		have_gpu_input = new_val;
	}

	void set_GPUOutput(const bool new_val) {
		if (have_gpu_output == new_val) {
			return;
		}
		// TODO: Reallocate buffers...
		have_gpu_output = new_val;
	}

	/*
	 * This object may not be copied nor moved.
	 */
	MapGPU_Node(const MapGPU_Node &) = delete;
	MapGPU_Node(MapGPU_Node &&) = delete;
	MapGPU_Node &operator=(const MapGPU_Node &) = delete;
	MapGPU_Node &operator=(MapGPU_Node &&) = delete;
};

} // namespace wf

#endif
