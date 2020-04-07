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
#include <queue>
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
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i]);
	}
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
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i], result_buffer[i]);
	}
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
template<typename tuple_t, typename func_t>
__global__ void
run_map_kernel_keyed_ip(func_t map_func, tuple_t *tuple_buffer,
			char **scratchpads, const std::size_t scratchpad_size,
			const std::size_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i], scratchpads[i], scratchpad_size);
	}
}

template<typename tuple_t, typename result_t, typename func_t>
__global__ void
run_map_kernel_keyed_nip(func_t map_func, tuple_t *tuple_buffer,
			 result_t *result_buffer, char **scratchpads,
			 const std::size_t scratchpad_size,
			 const std::size_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		map_func(tuple_buffer[i], result_buffer[i], scratchpads[i],
			 scratchpad_size);
	}
}

template<typename tuple_t, typename result_t, typename func_t, typename closing_func_t>
class MapGPU_Node: public ff::ff_node_t<tuple_t, result_t>
{
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
	static constexpr bool is_not_in_place_keyless =
		is_invocable<F, const tuple_t &, tuple_t &>::value;

	template<typename F>
	static constexpr bool is_in_place_keyed =
		is_invocable<F, tuple_t &, char *, std::size_t>::value;

	template<typename F>
	static constexpr bool is_not_in_place_keyed =
		is_invocable<F, const tuple_t &, result_t &, char *, std::size_t>::value;

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
	 * Class memebers
	 */
	func_t map_func;
	closing_func_t closing_func;
	RuntimeContext context;

	/*
	 * This struct, one per individual key, contains a queue of tuples with
	 * the same key and a pointer to the respective memory area to be
	 * used as scratchpad, to save state.
	 */
	struct KeyControlBlock
	{
		std::queue<tuple_t *> queue;
		char *scratchpad;
	};
	std::unordered_map<int, KeyControlBlock> key_control_block_map;

	std::string operator_name;
	int total_buffer_capacity;
	int gpu_threads_per_block;
	int gpu_blocks;

	int current_buffer_capacity {0};

	cudaStream_t cuda_stream;
	tuple_t *cpu_tuple_buffer;
	tuple_t *gpu_tuple_buffer;
	result_t *cpu_result_buffer;
	result_t *gpu_result_buffer;

	/*
	 * Scratchpad buffers are used to store pointers to individual
	 * scratchpads.  It is passed to the kernel so that it knows where to
	 * find the corresponding scratchpad for each processed tuple.
	 */
	char **cpu_scratchpad_buffer;
	char **gpu_scratchpad_buffer;
	std::size_t scratchpad_size;
	bool was_batch_started {false};

#if defined(TRACE_WINDFLOW)
	unsigned long rcvTuples {0};
	double avg_td_us {0};
	double avg_ts_us {0};
	volatile unsigned long startTD, startTS, endTD, endTS;
	std::ofstream *logfile = nullptr;
#endif

	template<typename F=func_t, typename std::enable_if_t<is_in_place<F>, int> = 0>
	void setup_gpu_result_buffer() { gpu_result_buffer = gpu_tuple_buffer; }

	template<typename F=func_t, typename std::enable_if_t<!is_in_place<F>, int> = 0>
	void
	setup_gpu_result_buffer()
	{
		const auto size = total_buffer_capacity * sizeof(result_t);
		if (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU result buffer");
		}
	}

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	void setup_scratchpad_buffers() {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void
	setup_scratchpad_buffers()
	{
		const auto size = total_buffer_capacity * sizeof(char *);
		if (cudaMallocHost(&cpu_scratchpad_buffer, size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate CPU scratchpad buffer");
		}
		if (cudaMalloc(&gpu_scratchpad_buffer, size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU scratchpad buffer");
		}
	}

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	result_t *
	svc_aux(tuple_t *t)
	{
#if defined (TRACE_WINDFLOW)
		startTS = current_time_nsecs();
		if (rcvTuples == 0)
			startTD = current_time_nsecs();
		rcvTuples++;
#endif
		cpu_tuple_buffer[current_buffer_capacity] = *t;
		++current_buffer_capacity;
		delete t;
		if (current_buffer_capacity < total_buffer_capacity) {
			return this->GO_ON;
		}
		send_last_batch_if_any();
		const auto size = total_buffer_capacity * sizeof(tuple_t);
		cudaMemcpy(gpu_tuple_buffer, cpu_tuple_buffer, size,
			   cudaMemcpyHostToDevice);
		current_buffer_capacity = 0;
		run_kernel();
		was_batch_started = true; //TODO: Redundant after the first
					  //time! Could be made more
					  //efficient,but uglier... (At least
					  //it's not a branch).
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

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	result_t *
	svc_aux(tuple_t *t)
	{
		if (current_buffer_capacity < total_buffer_capacity) {
			const auto &key = t->key;
			if (key_control_block_map.find(key) == key_control_block_map.end()) {
				auto &scratchpad = key_control_block_map[key].scratchpad;
				if (cudaMalloc(&scratchpad, scratchpad_size) != cudaSuccess) {
					failwith("MapGPU_Node failed to allocate GPU scratchpad for key "
						 + std::to_string(key));
				}
				++current_buffer_capacity;
			}
			key_control_block_map[key].queue.push(t);
			return this->GO_ON;
		}
		send_last_batch_if_any();

		auto kcb_iter = key_control_block_map.begin();
		for (auto i = 0; i < total_buffer_capacity; ++i) {
			while (kcb_iter->second.queue.empty()) {
				++kcb_iter;
			}
			const auto t = kcb_iter->second.queue.front();
			cpu_tuple_buffer[i] = *t;
			delete t;
			kcb_iter->second.queue.pop();
			cpu_scratchpad_buffer[i] = kcb_iter->second.scratchpad;
			if (kcb_iter->second.queue.empty()) {
				--current_buffer_capacity;
			}
			++kcb_iter;
		}
		const auto size = total_buffer_capacity * sizeof(char *);
		cudaMemcpy(gpu_scratchpad_buffer, cpu_scratchpad_buffer, size,
			   cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tuple_buffer, cpu_tuple_buffer, size,
			   cudaMemcpyHostToDevice);
		run_kernel();
		was_batch_started = true;
		return this->GO_ON;
	}

	void
	send_last_batch_if_any()
	{
		if (!was_batch_started) {
			return;
		}
		cudaStreamSynchronize(cuda_stream);
		const auto size = total_buffer_capacity * sizeof(result_t);
		cudaMemcpy(cpu_result_buffer, gpu_result_buffer, size,
			   cudaMemcpyDeviceToHost);
		for (auto i = 0; i < total_buffer_capacity; ++i) {
			this->ff_send_out(new result_t {cpu_result_buffer[i]});
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyless<F>, int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, total_buffer_capacity);
	}


	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyless<F>, int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 total_buffer_capacity);
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyed<F>, int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_keyed_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_scratchpad_buffer,
			 scratchpad_size, total_buffer_capacity);
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyed<F>, int> = 0>
	void
	run_kernel()
	{
		run_map_kernel_keyed_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
			(map_func, gpu_tuple_buffer, gpu_result_buffer,
			 gpu_scratchpad_buffer, scratchpad_size, total_buffer_capacity);
	}

	/*
	 * Ulness the stream happens to have exactly a number of tuples that is
	 * a multiple of the buffer capacity, some tuples will be "left out" at
	 * the end of the stream, since there aren't any tuples left to
	 * completely fill the buffer.  We compute the function on the remaining
	 * tuples directly on the CPU, for simplicity, since they are likely to
	 * be a much smaller number than the total stream length.
	 */
	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyless<F>, int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto &t = cpu_tuple_buffer[i];
			map_func(t);
			this->ff_send_out(new tuple_t {t});
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyless<F>, int> = 0>
	void
	process_last_tuples()
	{
		for (auto i = 0; i < current_buffer_capacity; ++i) {
			auto res = new result_t {};
			map_func(cpu_tuple_buffer[i], *res);
			this->ff_send_out(res);
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place_keyed<F>, int> = 0>
	void
	process_last_tuples()
	{
		std::vector<char> cpu_scratchpad(scratchpad_size);
		for (auto &kv : key_control_block_map) {
			auto &queue = kv.second.queue;
			auto &gpu_scratchpad = kv.second.scratchpad;

			cudaMemcpy(cpu_scratchpad.data(), gpu_scratchpad,
				   cpu_scratchpad.size(), cudaMemcpyDeviceToHost);
			for (; !queue.empty(); queue.pop()) {
				auto t = queue.front();
				map_func(*t, cpu_scratchpad.data(), cpu_scratchpad.size());
				this->ff_send_out(t); // Can re-send the same tuple.
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_not_in_place_keyed<F>, int> = 0>
	void
	process_last_tuples()
	{
		std::vector<char> cpu_scratchpad(scratchpad_size);

		for (auto &kv : key_control_block_map) {
			auto &queue = kv.second.queue;
			auto &gpu_scratchpad = kv.second.scratchpad;

			cudaMemcpy(cpu_scratchpad.data(), gpu_scratchpad,
				   cpu_scratchpad.size(), cudaMemcpyDeviceToHost);
			for (; !queue.empty(); queue.pop()) {
				auto t = queue.front();
				auto res = new result_t {};

				map_func(*t, *res, cpu_scratchpad.data(), cpu_scratchpad.size());
				delete t;
				this->ff_send_out(res);
			}
		}
	}

	template<typename F=func_t, typename std::enable_if_t<is_in_place<F>, int> = 0>
	void deallocate_gpu_result_buffer() {}

	template<typename F=func_t, typename std::enable_if_t<!is_in_place<F>, int> = 0>
	void deallocate_gpu_result_buffer() { cudaFree(gpu_result_buffer); }

	template<typename F=func_t, typename std::enable_if_t<!is_keyed<F>, int> = 0>
	void deallocate_scratchpad_buffers() {}

	template<typename F=func_t, typename std::enable_if_t<is_keyed<F>, int> = 0>
	void
	deallocate_scratchpad_buffers()
	{
		for (auto &kcb : key_control_block_map) {
			cudaFree(kcb.second.scratchpad);
		}
		cudaFreeHost(cpu_scratchpad_buffer);
		cudaFree(gpu_scratchpad_buffer);
	}

public:
	/*
	 * A single worker allocates both CPU and GPU buffers to store tuples.
	 * In case the function to be used is NOT in-place, it also allocates
	 * enough space for the CPU and GPU buffers to store the results.  The
	 * first constructor is used for keyless (stateless) version, the second is
	 * for the keyed (stateful) version.
	 */
	MapGPU_Node(func_t map_func, std::string name, RuntimeContext context,
		    int total_buffer_capacity, int gpu_threads_per_block,
		    closing_func_t closing_func)
		: MapGPU_Node {map_func, name, context, total_buffer_capacity,
			       gpu_threads_per_block, 0, closing_func}
	{}

	MapGPU_Node(func_t map_func, std::string name, RuntimeContext context,
		    int total_buffer_capacity, int gpu_threads_per_block,
		    int scratchpad_size, closing_func_t closing_func)
		: map_func {map_func}, operator_name {name}, context {context},
		  total_buffer_capacity {total_buffer_capacity},
		  gpu_threads_per_block {gpu_threads_per_block},
		  gpu_blocks {std::ceil(total_buffer_capacity
					/ float {gpu_threads_per_block})},
		  scratchpad_size {scratchpad_size},
		  closing_func {closing_func}
	{
		const auto tuple_size = sizeof(tuple_t) * total_buffer_capacity;
		const auto result_size = sizeof(result_t) * total_buffer_capacity;

		if (cudaMallocHost(&cpu_tuple_buffer, tuple_size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate CPU tuple buffer");
		}
		if (cudaMallocHost(&cpu_result_buffer, result_size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate CPU result buffer");
		}
		if (cudaMalloc(&gpu_tuple_buffer, tuple_size) != cudaSuccess) {
			failwith("MapGPU_Node failed to allocate GPU tuple buffer");
		}
		if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
			failwith("cudaStreamCreate() failed in MapGPU_Node");
		}
		setup_gpu_result_buffer();
		setup_scratchpad_buffers();
	}

	~MapGPU_Node()
	{
		cudaFreeHost(cpu_tuple_buffer);
		cudaFreeHost(cpu_result_buffer);
		cudaFree(gpu_tuple_buffer);
		deallocate_gpu_result_buffer();
		deallocate_scratchpad_buffers();
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

	/*
	 * svc function used by the FastFlow runtime.  It calls the appropriate
	 * auxiliary function based on whether the function is stateless or not
	 * (keyed).
	 */
	result_t *svc(tuple_t *t) { return svc_aux(t); }

	/*
	 * Acts on receiving the EOS (End Of Stream) signal by the FastFlow
	 * runtime.  It computes the last remaining tuples on the CPU, for
	 * simplicity, then sends out any remaining results from the last CUDA
	 * kernel.
	 */
	void
	eosnotify(ssize_t)
	{
		send_last_batch_if_any();
		process_last_tuples();
	}

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
