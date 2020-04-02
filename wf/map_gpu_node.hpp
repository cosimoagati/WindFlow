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

#include <cmath>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>

#include <ff/node.hpp>
#include "basic.hpp"
#include "standard_nodes.hpp"
#include "map_gpu_utils.hpp"

namespace wf
{
// N.B.: CUDA __global__ kernels must not be member functions.
// TODO: can we make the distinction simpler, with less repetition?
/**
 * \brief Map kernel (in-place version). Run function and store results in same
 * buffer
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer pointer to the start of the buffered tuples
 * (memory area accessible to GPU)
 * \param buffer_capacity How many tuples the buffer contains.
 */
template<typename tuple_t, typename func_t>
__global__ void
run_map_kernel_ip(func_t map_func, tuple_t *tuple_buffer,
		  const std::size_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride)
		map_func(tuple_buffer[i]);
}

/**
 * \brief Map kernel (non-in-place version). Run function and store results a new buffer
 * \param map_func The function to be computed on each tuple.
 * \param tuple_buffer pointer to the start of the buffered tuples
 * \param result_buffer pointer to the start of the buffer that will contain the results.
 * (memory area accessible to GPU)
 * \param buffer_capacity How many tuples the buffer contains.
 */
template<typename tuple_t, typename result_t, typename func_t>
__global__ void
run_map_kernel_nip(func_t map_func, tuple_t *tuple_buffer,
		   result_t *result_buffer, const std::size_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride)
		map_func(tuple_buffer[i], result_buffer[i]);
}

/*
 * The following two are keyed versions of the kernel.  They should work on
 * tuples all of different keys.  The provided map function accesses an internal
 * state, saved in a dedicated scratchpad, which the current tuple updates or
 * makes use of.  This state is allocated on the GPU memory, therefore merely
 * defined by a location and a size.  All keys have a separate scratchpad, which
 * are all stored and indexed from a single data structure, using the tuple's
 * own key.
 */
template<typename tuple_t, typename func_t, typename int_t=std::size_t>
__global__ void
run_map_kernel_keyed_ip(func_t map_func, tuple_t *tuple_buffer,
			char *scratchpads, const int_t scratchpad_size,
			const int_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		const auto key = tuple_buffer[i].key;
		map_func(tuple_buffer[i], &scratchpads[key], scratchpad_size);
	}
}

template<typename tuple_t, typename result_t, typename func_t>
__global__ void
run_map_kernel_keyed_nip(func_t map_func, tuple_t *tuple_buffer,
			 result_t *result_buffer, char *scratchpads,
			 const std::size_t scratchpad_size,
			 const std::size_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		const auto key = tuple_buffer[i].key;
		map_func(tuple_buffer[i], result_buffer[i],
			 &scratchpads[key], scratchpad_size);
	}
}

template<typename tuple_t, typename result_t, typename func_t,
	 typename closing_func_t, typename routing_func_t>
class MapGPU_Node: public ff::ff_node_t<tuple_t, result_t>
{
	/*
	 * Perform a compile-time check in order to make sure the function to
	 * be computed by the MapGPU operator has a valid signature.
	 */
	static_assert((is_invocable<func_t, tuple_t &>::value
		       && !is_invocable<func_t, const tuple_t &, result_t &>::value
		       && !is_invocable<func_t, tuple_t &, char *, std::size_t>::value
		       && !is_invocable<func_t, const tuple_t &, char *, std::size_t>::value)
		      || (!is_invocable<func_t, tuple_t &>::value
			  && is_invocable<func_t, const tuple_t &, result_t &>::value
			  && !is_invocable<func_t, tuple_t &, char *, std::size_t>::value
			  && !is_invocable<func_t, const tuple_t &, char *, std::size_t>::value)
		      || (!is_invocable<func_t, tuple_t &>::value
			  && !is_invocable<func_t, const tuple_t &, result_t &>::value
			  && is_invocable<func_t, tuple_t &, char *, std::size_t>::value
			  && !is_invocable<func_t, const tuple_t &, char *, std::size_t>::value)
		      || (!is_invocable<func_t, tuple_t &>::value
			  && !is_invocable<func_t, const tuple_t &, result_t &>::value
			  && !is_invocable<func_t, tuple_t &, char *, std::size_t>::value
			  && is_invocable<func_t, const tuple_t &, char *, std::size_t>::value),
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
	static_assert(!(is_invocable<func_t, tuple_t &>::value
			|| is_invocable<func_t, tuple_t &, char *, std::size_t>::value)
		      || std::is_same<tuple_t, result_t>::value,
		      "WindFlow Error: if instantiating MapGPU with an in-place "
		      "function, the input type and the output type must match.");

	func_t map_func; // The function to be computed on the tuples. May be in
			 // place or not.
	closing_func_t closing_func;
	std::string name; // string of the unique name of the operator
	RuntimeContext context; // RuntimeContext
	int tuple_buffer_capacity;
	int gpu_threads_per_block;
	int gpu_blocks;
	int currently_buffered_tuples {0};

	cudaStream_t cuda_stream;
	tuple_t *cpu_tuple_buffer;
	tuple_t *gpu_tuple_buffer;
	result_t *cpu_result_buffer;
	result_t *gpu_result_buffer;

	char *scratchpads;
	int number_of_keys;
	std::size_t scratchpad_size;

	// Used to avoid sending results for the first batch.
	bool was_first_batch_sent {false};

#if defined(TRACE_WINDFLOW)
	unsigned long rcvTuples {0};
	double avg_td_us {0};
	double avg_ts_us {0};
	volatile unsigned long startTD, startTS, endTD, endTS;
	std::ofstream *logfile = nullptr;
#endif
	/*
	 * Through the use of std::enable_if, we define different variants of
	 * behavior that is different between the in-place and the non-in-place
	 * versions of the MapGPU operator.  Only the desired version will be
	 * compiled.
	 */

	/*
	 * If the function is in place, we can conveniently refer to the CPU and
	 * GPU tuple buffers as if they were the result buffer, since they play
	 * both roles.
	 */
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &>::value
					   || is_invocable<F, tuple_t &,
							   char *,
							   std::size_t>::value,
					   int> = 0>
	void
	setup_result_buffer()
	{
		cpu_result_buffer = cpu_tuple_buffer;
		gpu_result_buffer = gpu_tuple_buffer;
	}

	/*
	 * If the function is not in place, result buffers are distinct and must
	 * be allocated.
	 */
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, const tuple_t &,
							result_t &>::value
					   || is_invocable<F, const tuple_t &,
							   result_t &, char *,
							   std::size_t>::value,
					   int> = 0>
	void
	setup_result_buffer()
	{
		const auto size = sizeof(result_t) * tuple_buffer_capacity;
		if (cudaMallocHost(&cpu_result_buffer, size) != cudaSuccess)
			failwith("MapGPU_Node failed to allocate CPU result buffer");
		if (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess)
			failwith("MapGPU_Node failed to allocate GPU result buffer");
	}

	/*
	  i		 * When all tuples have been buffered, it's time to feed them
	  * to the CUDA kernel.
	  */
	// In-place keyless version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &>::value,
					   int> = 0>
	void
	run_kernel()
	{
		// TODO: the only reason these functions are used is to call a
		// different kernel!  Can we do some template magic with the
		// kernels themselves in to remove these functions?
		run_map_kernel_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer,
			 tuple_buffer_capacity);
	}

	// Non in-place keyless version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F,
							const tuple_t &,
							result_t &>::value,
					   int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 tuple_buffer_capacity);
	}

	// In-place keyed version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &,
							char *, std::size_t>::value,
					   int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_keyed_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, scratchpads,
			 scratchpad_size, tuple_buffer_capacity);
	}

	// Non in-place keyed version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, const tuple_t &, result_t &,
							char *, std::size_t>::value,
					   int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_keyed_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 scratchpads, scratchpad_size, tuple_buffer_capacity);
	}

	/*
	 * Ulness the stream happens to have exactly a number of tuples that is
	 * a multiple of the buffer capacity, some tuples will be "left out" at
	 * the end of the stream, since there aren't any tuples left to
	 * completely fill the buffer.  We compute the function on the remaining
	 * tuples directly on the CPU, for simplicity, since they are likely to
	 * be a much smaller number than the total stream length.
	 */
	// In-place keyless version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &>::value, int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < currently_buffered_tuples; ++i) {
			auto &t = cpu_tuple_buffer[i];
			map_func(t);
			this->ff_send_out(new tuple_t {t});
		}
	}

	// Non in-place keyless version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, const tuple_t &,
							result_t &>::value,
					   int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < currently_buffered_tuples; ++i) {
			const auto &t = cpu_tuple_buffer[i];
			auto &res = cpu_result_buffer[i];
			map_func(t, res);
			this->ff_send_out(new result_t {res});
		}
	}

	// In-place keyed version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &,
							char *, std::size_t>::value,
					   int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < currently_buffered_tuples; ++i) {
			auto &t = cpu_tuple_buffer[i];
			map_func(t, scratchpads[t.key], scratchpad_size);
			this->ff_send_out(new tuple_t {t});
		}
	}

	// Non in-place keyed version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, const tuple_t &, result_t &,
							char *, std::size_t>::value,
					   int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < currently_buffered_tuples; ++i) {
			const auto &t = cpu_tuple_buffer[i];
			auto &res = cpu_result_buffer[i];
			map_func(t, res, scratchpads[t.key],
				 scratchpad_size);
			this->ff_send_out(new result_t {res});
		}
	}

	// Do nothing if the function is in place.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, tuple_t &>::value,
					   int> = 0>
	void deallocate_result_buffer() {}

	// Non in-place version.
	template<typename F=func_t,
		 typename std::enable_if_t<is_invocable<F, const tuple_t &,
							result_t &>::value,
					   int> = 0>
	void
	deallocate_result_buffer()
	{
		cudaFreeHost(cpu_result_buffer);
		cudaFree(gpu_result_buffer);
	}

public:
	/*
	 * A single worker allocates both CPU and GPU buffers to store tuples.
	 * In case the function to be used is NOT in-place, it also allocates
	 * enough space for the CPU and GPU buffers to store the results.  The
	 * first constructor is used for keyed (stateful) version, the second is
	 * for the keyless (stateless) version.
	 */
	MapGPU_Node(func_t func, std::string name, RuntimeContext context,
		    int tuple_buffer_capacity, int gpu_threads_per_block,
		    char *scratchpads, int number_of_keys, int scratchpad_size,
		    closing_func_t closing_func)
		: map_func {func}, name {name}, context {context},
		  tuple_buffer_capacity {tuple_buffer_capacity},
		  		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(tuple_buffer_capacity
					/ float {gpu_threads_per_block})},
		  scratchpads {scratchpads},
		  closing_func {closing_func}
	{
		const auto size = sizeof(tuple_t) * tuple_buffer_capacity;

		if (cudaMallocHost(&cpu_tuple_buffer, size) != cudaSuccess)
			failwith("MapGPU_Node failed to allocate CPU tuple buffer");
		if (cudaMalloc(&gpu_tuple_buffer, size) != cudaSuccess)
			failwith("MapGPU_Node failed to allocate GPU tuple buffer");
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess)
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		setup_result_buffer();
	}

	MapGPU_Node(func_t func, std::string name, RuntimeContext context,
		    int tuple_buffer_capacity, int gpu_threads_per_block,
		    closing_func_t closing_func)
		: MapGPU_Node {func, name, context, tuple_buffer_capacity,
			       gpu_threads_per_block, nullptr, 0, 0,
			       closing_func}
	{}

	~MapGPU_Node()
	{
		cudaFreeHost(cpu_tuple_buffer);
		cudaFree(gpu_tuple_buffer);
		deallocate_result_buffer();
		cudaStreamDestroy(cuda_stream);
	}

	// svc_init method (utilized by the FastFlow runtime)
	int
	svc_init()
	{
#if defined(TRACE_WINDFLOW)
		logfile = new std::ofstream();
		name += "_node_" + std::to_string(ff::ff_node_t<tuple_t,
						  result_t>::get_my_id())
			+ ".log";
		std::string filename =
			std::string(STRINGIFY(TRACE_WINDFLOW)) + "/"
			+ name;
		logfile->open(filename);
#endif
		return 0;
	}

	// svc method (utilized by the FastFlow runtime)
	/*
	 * After buffering enough tuples, send the previously computed results
	 * (except for the first time), then start the CUDA kernel on the newly
	 * buffered elements.
	 */
	result_t *
	svc(tuple_t *t)
	{
#if defined (TRACE_WINDFLOW)
		startTS = current_time_nsecs();
		if (rcvTuples == 0)
			startTD = current_time_nsecs();
		rcvTuples++;
#endif
		if (currently_buffered_tuples < tuple_buffer_capacity) {
			cpu_tuple_buffer[currently_buffered_tuples] = *t;
			++currently_buffered_tuples;
			delete t;
			return this->GO_ON;
		}
		if (!was_first_batch_sent) {
			was_first_batch_sent = true;
			return this->GO_ON;
		}
		cudaStreamSynchronize(cuda_stream);
		cudaMemcpy(cpu_result_buffer, gpu_result_buffer,
			   tuple_buffer_capacity * sizeof(result_t),
			   cudaMemcpyDeviceToHost);

		for (auto i = 0; i < tuple_buffer_capacity; ++i)
			this->ff_send_out(new result_t {cpu_result_buffer[i]});
		currently_buffered_tuples = 0;

		cudaMemcpy(gpu_tuple_buffer, cpu_tuple_buffer,
			   tuple_buffer_capacity * sizeof(tuple_t),
			   cudaMemcpyHostToDevice);
		run_kernel();
#if defined(TRACE_WINDFLOW)
		endTS = current_time_nsecs();
		endTD = current_time_nsecs();
		double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
		avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
		double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
		avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
		startTD = current_time_nsecs();
#endif
		return this->GO_ON;
	}

	/*
	 * Acts on receiving the EOS (End Of Stream) signal by the FastFlow
	 * runtime.  Based on whether the map function is in place or not, it
	 * calls a different process_last_tuples method.
	 */
	// TODO: Why can't we use std::enable_if directly with this
	// function?
	void eosnotify(ssize_t) { process_last_tuples(); }

	// svc_end method (utilized by the FastFlow runtime)
	void
	svc_end()
	{
		closing_func(context);
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
