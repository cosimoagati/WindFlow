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
 *  @brief Map_GPU operator executing a one-to-one transformation on the input stream
 *
 *  @section Map_GPU (Description)
 *
 *  This file implements the Map operator able to execute a one-to-one transformation
 *  on each tuple of the input data stream. The transformation should be stateless and
 *  must produce one output result for each input tuple consumed.
 *
 *  The template parameters tuple_t and result_t must be default constructible, with a
 *  copy Constructor and copy assignment operator, and they must provide and implement
 *  the setControlFields() and getControlFields() methods.
 */

#ifndef MAP_GPU_H
#define MAP_GPU_H

#include <cstdlib>
#include <deque>
#include <string>
#include <vector>
#include <unordered_map>

#include <ff/farm.hpp>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp"

namespace wf
{

/*
 * Reimplementation of std::is_invocable, unfortunately needed
 * since CUDA doesn't yet support C++17
 */
// TODO: Move this in some default utilities header.
template<typename F, typename... Args>
struct is_invocable :
		std::is_constructible<std::function<void(Args ...)>,
				      std::reference_wrapper<typename std::remove_reference<F>::type>>
{};

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
template<typename tuple_t, typename func_t, typename int_t=std::size_t>
__global__ void
run_map_kernel_ip(func_t map_func, tuple_t *tuple_buffer,
		  const int_t buffer_capacity)
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
template<typename tuple_t, typename result_t, typename func_t,
	 typename int_t=std::size_t>
__global__ void
run_map_kernel_nip(func_t map_func, tuple_t *tuple_buffer,
		   result_t *result_buffer, const int_t buffer_capacity)
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

template<typename tuple_t, typename result_t, typename func_t,
	 typename int_t=std::size_t>
__global__ void
run_map_kernel_keyed_nip(func_t map_func, tuple_t *tuple_buffer,
			 result_t *result_buffer, char *scratchpads,
			 const int_t scratchpad_size,
			 const int_t buffer_capacity)
{
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	const auto stride = blockDim.x * gridDim.x;
	for (auto i = index; i < buffer_capacity; i += stride) {
		const auto key = tuple_buffer[i].key;
		map_func(tuple_buffer[i], result_buffer[i],
			 &scratchpads[key], scratchpad_size);
	}
}

// TODO: This should be included in some utils file, it doesn't belong to
// the MapGPU class...
// TODO: Should we always exit with an error or throw an exception?
// Exiting with an error allows the library to be compiled with exceptions
// disabled.
/**
 * \brief Prints error on the screen and exits.
 * \param err The error to be shown.
 */
inline void
failwith(const std::string &err)
{
	std::cerr << RED << "WindFlow Error: " << err << DEFAULT_COLOR
		  << std::endl;
	std::exit(EXIT_FAILURE);
}

template<typename int_t=std::size_t>
inline void
check_constructor_parameters(int_t pardegree, int_t max_buffered_tuples,
			     int_t gpu_blocks, int_t gpu_threads_per_block)
{
	static_assert(std::is_signed<int_t>::value,
		      "WindFlow Error: integer type used for parameters should "
		      "be signed, can't check for validity otherwise.");
	if (pardegree <= 0)
		failwith("MapGPU has non-positive parallelism");
	if (max_buffered_tuples <= 0)
		failwith("MapGPU has non-positive maximum buffered tuples");
	if (gpu_blocks <= 0)
		failwith("MapGPU has non-positive number of GPU blocks");
	if (gpu_threads_per_block <= 0)
		failwith("MapGPU has non-positive number of "
			 "GPU threads per block");
}

/**
 *  \class MapGPU
 *
 *  \brief Map operator executing a one-to-one transformation on the input
 *  stream, GPU version
 *
 *  This class implements the Map operator executing a one-to-one stateless
 *  transformation on each tuple of the input stream.
 */
template<typename tuple_t, typename result_t, typename func_t>
class MapGPU: public ff::ff_farm
{
public:
	/*
	 * Performs a compile-time check in order to make sure the function to
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

	/// type of the closing function
	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t,
							 std::size_t)>;
private:
	static constexpr auto default_max_buffered_tuples = 256;
	static constexpr auto default_gpu_blocks = 1;
	static constexpr auto default_gpu_threads_per_block = 256;

	// Arbitrary default values for now.
	static constexpr auto NUMBER_OF_KEYS = 256;
	static constexpr auto SCRATCHPAD_SIZE = 64; // Size of a single
						    // scratchpad in chars.
	char *scratchpads {nullptr}; // Points to the GPU memory area containing
				     // the scratchpads for stateful keyed
				     // functions.
	bool is_keyed; // is the MapGPU is configured with keyBy or not?
	bool used; // is the MapGPU used in a MultiPipe or not?

	// friendships with other classes in the library
	friend class MultiPipe;

	class MapGPU_Node: public ff::ff_node_t<tuple_t, result_t>
	{
		func_t map_func; // The function to be computed on the
				 // tuples. May be in place or not.
		closing_func_t closing_func;
		std::string name; // string of the unique name of the operator
		RuntimeContext context; // RuntimeContext
		int max_buffered_tuples;
		int gpu_blocks;
		int gpu_threads_per_block;

		cudaStream_t cuda_stream;
		std::vector<tuple_t> cpu_tuple_buffer;
		tuple_t *gpu_tuple_buffer;
		std::vector<result_t> cpu_result_buffer;
		result_t *gpu_result_buffer;

		char *scratchpads;
		int number_of_keys;
		std::size_t scratchpad_size;

#if defined(TRACE_WINDFLOW)
		unsigned long rcvTuples {0};
		double avg_td_us {0};
		double avg_ts_us {0};
		volatile unsigned long startTD, startTS, endTD, endTS;
		std::ofstream *logfile = nullptr;
#endif

		/*
		 * This function encapsulates behavior shared by all constructors.
		 */
		void
		init_node()
		{
			if (cudaMalloc(&gpu_tuple_buffer,
				       max_buffered_tuples * sizeof(tuple_t)) != cudaSuccess)
				failwith("MapGPU_Node failed to allocate GPU memory area");
			if (cudaStreamCreate(&cuda_stream) != cudaSuccess)
				failwith("cudaStreamCreate() failed in MapGPU_Node");
			setup_result_buffer(max_buffered_tuples * sizeof(result_t));
		}

		/*
		 * Through the use of std::enable_if, we define different
		 * variants of behavior that is different between the in-place
		 * and the non-in-place versions of the MapGPU operator.  Only
		 * the desired version will be compiled.
		 */
		template<typename int_t=size_t>
		void
		setup_result_buffer(typename std::enable_if_t<is_invocable<func_t, tuple_t &>::value
				    || is_invocable<func_t, tuple_t &, char *, std::size_t>::value,
				    int_t>)
		{
			cpu_result_buffer.reserve(0);
		}

		template<typename int_t=std::size_t>
		void
		setup_result_buffer(typename std::enable_if_t<is_invocable<func_t, const tuple_t &, result_t &>::value
				    || is_invocable<func_t, const tuple_t &, result_t & , char *, std::size_t>::value,
				    int_t> size)
		{
			// TODO: allocating all these tuples at the beginning is
			// not very elegant...
			cpu_result_buffer.resize(max_buffered_tuples);
			if (cudaMalloc(&gpu_result_buffer, size) != cudaSuccess)
				failwith("MapGPU_Node failed to allocate shared memory area");
		}

		/*
		 * When all tuples have been buffered, it's time to feed them
		 * to the CUDA kernel.
		 */
		// In-place keyless version.
		template<typename F=func_t>
		void
		process_buffered_tuples(typename std::enable_if_t<
					is_invocable<F, tuple_t &>::value, F>)
		{
			run_map_kernel_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
				(map_func, gpu_tuple_buffer,
				 max_buffered_tuples);
			cudaStreamSynchronize(cuda_stream);
			cudaMemcpy(cpu_tuple_buffer.data(), gpu_tuple_buffer,
				   max_buffered_tuples * sizeof(tuple_t),
				   cudaMemcpyDeviceToHost);
			for (const auto &t: cpu_tuple_buffer) {
				this->ff_send_out(reinterpret_cast<result_t *>
						  (new tuple_t {t}));
			}
			cpu_tuple_buffer.clear();
		}

		// Non in-place keyless version.
		template<typename F=func_t>
		void
		process_buffered_tuples(typename std::enable_if_t<
					is_invocable<F, const tuple_t &, result_t &>::value,
					F>)
		{
			run_map_kernel_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
				(map_func, gpu_tuple_buffer,
				 gpu_result_buffer, max_buffered_tuples);
			cudaStreamSynchronize(cuda_stream);
			cudaMemcpy(cpu_result_buffer.data(), gpu_result_buffer,
				   max_buffered_tuples * sizeof(result_t),
				   cudaMemcpyDeviceToHost);
			for (const auto &t : cpu_result_buffer)
				this->ff_send_out(new result_t {t});
			cpu_tuple_buffer.clear();
		}

		// TODO: Why do I get redeclaration error? These should be
		// ignored since std::enable_if fails and does not have a type.
		// Answer: if more than one template fails to substitute, they
		// are considered the exact same template even if they look
		// different!

		// In-place keyed version.
		template<typename int_t=std::size_t>
		void
		process_buffered_tuples(typename std::enable_if_t<
					is_invocable<func_t, tuple_t &, char *, int_t>::value,
					func_t>)
		{
			run_map_kernel_keyed_ip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
				(map_func, gpu_tuple_buffer, scratchpads,
				 scratchpad_size, max_buffered_tuples);
			cudaStreamSynchronize(cuda_stream);
			cudaMemcpy(cpu_tuple_buffer.data(), gpu_tuple_buffer,
				   max_buffered_tuples * sizeof(tuple_t),
				   cudaMemcpyDeviceToHost);
			for (const auto &t: cpu_tuple_buffer) {
				this->ff_send_out(reinterpret_cast<result_t *>
						  (new tuple_t {t}));
			}
			cpu_tuple_buffer.clear();
		}

		// Non in-place keyed version.
		template<typename int_t=std::size_t>
		void
		process_buffered_tuples(typename std::enable_if_t<
					is_invocable<func_t, const tuple_t &, result_t &, char *, int_t>::value,
					func_t>)
		{
			run_map_kernel_keyed_nip<<<gpu_blocks, gpu_threads_per_block, 0, cuda_stream>>>
				(map_func, gpu_tuple_buffer,
				 gpu_result_buffer, scratchpads,
				 scratchpad_size, max_buffered_tuples);
			cudaStreamSynchronize(cuda_stream);
			cudaMemcpy(cpu_result_buffer.data(), gpu_result_buffer,
				   max_buffered_tuples * sizeof(result_t),
				   cudaMemcpyDeviceToHost);
			for (const auto &t : cpu_result_buffer)
				this->ff_send_out(new result_t {t});
			cpu_tuple_buffer.clear();
		}

		/* 
		 * Ulness the stream happens to have exactly a number of tuples
		 * that is a multiple of the buffer capacity, some tuples will
		 * be "left out" at the end of the stream, since there aren't
		 * any tuples left to completely fill the buffer.  We compute
		 * the function on the remaining tuples directly on the CPU, for
		 * simplicity, since they are likely to be a much smaller number
		 * than the total stream length.
		 */
		// In-place keyless version.
		template<typename F=func_t>
		void
		process_last_tuples(typename std::enable_if_t<
				    is_invocable<F, tuple_t &>::value, F>)
		{
			for (auto &t : cpu_tuple_buffer) {
				map_func(t);
				this->ff_send_out(reinterpret_cast<result_t *>
						  (new tuple_t {t}));
			}
		}

		// Non in-place keyless version.
		template<typename F=func_t>
		void
		process_last_tuples(typename std::enable_if_t<
				    is_invocable<F, const tuple_t &, result_t &>::value,
				    F>)
		{
			for (auto i = 0; i < cpu_tuple_buffer.size(); ++i) {
				const auto &t = cpu_tuple_buffer[i];
				auto &res = cpu_result_buffer[i];
				map_func(t, res);
				this->ff_send_out(new result_t {res});
			}
		}

		// In-place keyed version.
		template<typename int_t=std::size_t>
		void
		process_last_tuples(typename std::enable_if_t<
				    is_invocable<func_t, tuple_t &, char *, int_t>::value,
				    func_t>)
		{
			for (auto &t : cpu_tuple_buffer) {
				map_func(t, scratchpads[t.key], scratchpad_size);
				this->ff_send_out(reinterpret_cast<result_t *>
						  (new tuple_t {t}));
			}
		}

		// Non in-place keyed version.
		template<typename int_t=std::size_t>
		void
		process_last_tuples(typename std::enable_if_t<
				    is_invocable<func_t, const tuple_t &, result_t &, char *, int_t>::value,
				    func_t>)
		{
			for (auto i = 0; i < cpu_tuple_buffer.size(); ++i) {
				const auto &t = cpu_tuple_buffer[i];
				auto &res = cpu_result_buffer[i];
				map_func(t, res, scratchpads[t.key],
					 scratchpad_size);
				this->ff_send_out(new result_t {res});
			}
		}

		// Do nothing if the function is in place.
		template<typename F=func_t>
		void
		deallocate_result_buffer(typename std::enable_if_t<
					 is_invocable<F, tuple_t &>::value, result_t *>)
		{}

		// Non in-place version.
		template<typename F=func_t>
		void
		deallocate_result_buffer(typename std::enable_if_t<
					 is_invocable<F, const tuple_t &, result_t &>::value,
					 result_t *> buffer)
		{
			cudaFree(buffer);
		}

	public:
		/*
		 * A single worker allocates both CPU and GPU buffers to store
		 * tuples.  In case the function to be used is NOT in-place, it
		 * also allocates enough space for the CPU and GPU buffers to
		 * store the results.
		 * The first constructor is used for keyless (stateless)
		 * version, the second is for the keyed version.
		 */
		template<typename string_t=std::string, typename int_t=int>
		MapGPU_Node(func_t func, string_t name, RuntimeContext context,
			    int_t max_buffered_tuples, int_t gpu_blocks,
			    int_t gpu_threads_per_block,
			    closing_func_t closing_func)
			: map_func {func}, name {name}, context {context},
			  max_buffered_tuples {max_buffered_tuples},
			  gpu_blocks {gpu_blocks},
			  gpu_threads_per_block {gpu_threads_per_block},
			  closing_func {closing_func}
		{
			init_node();
		}

		template<typename string_t=std::string, typename int_t=int>
		MapGPU_Node(func_t func, string_t name, RuntimeContext context,
			    int_t max_buffered_tuples, int_t gpu_blocks,
			    int_t gpu_threads_per_block,
			    char *scratchpads,
			    int_t number_of_keys,
			    int_t scratchpad_size,
			    closing_func_t closing_func)
			: map_func {func}, name {name}, context {context},
			  max_buffered_tuples {max_buffered_tuples},
			  gpu_blocks {gpu_blocks},
			  gpu_threads_per_block {gpu_threads_per_block},
			  scratchpads {scratchpads},
			  closing_func {closing_func}
		{
			init_node();
		}

		~MapGPU_Node()
		{
			cudaFree(gpu_tuple_buffer);
			deallocate_result_buffer(gpu_result_buffer);
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
		result_t *
		svc(tuple_t *t)
		{
#if defined (TRACE_WINDFLOW)
			startTS = current_time_nsecs();
			if (rcvTuples == 0)
				startTD = current_time_nsecs();
			rcvTuples++;
#endif
			if (cpu_tuple_buffer.size() < max_buffered_tuples) {
				cpu_tuple_buffer.push_back(*t);
				delete t;
				return this->GO_ON;
			}
			cudaMemcpy(gpu_tuple_buffer, cpu_tuple_buffer.data(),
				   max_buffered_tuples * sizeof(tuple_t),
				   cudaMemcpyHostToDevice);
			process_buffered_tuples(map_func);
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
		 * Acts on receiving the EOS (End Of Stream) signal by the
		 * FastFlow runtime.  Based on whether the map function is in
		 * place or not, it calls a different process_last_tuples method.
		 */
		// TODO: Why can't we use std::enable_if directly with this
		// function?
		void
		eosnotify(ssize_t)
		{
			process_last_tuples(map_func);
		}

		// svc_end method (utilized by the FastFlow runtime)
		void
		svc_end()
		{
			// call the closing function
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
	};

public:
	/*
	 * Both constructors set the appropriate values, initialize the workers
	 * and start their internal farm.  The only difference they have is that
	 * one of them takes an additional parameter, the routing funciton.
	 * This is used in the keyed version.
	 */
	
	/**
	 *  \brief Constructor I
	 *
	 *  \param func function to be executed on each input tuple
	 *  \param pardegree parallelism degree of the MapGPU operator
	 *  \param name string with the unique name of the MapGPU operator
	 *  \param max_buffered_tuples numbers of tuples to buffer on the GPU
	 *  \param gpu_blocks the number of blocks to use when calling the GPU kernel
	 *  \param gpu_threads_per_block number of GPU threads per block
	 *  \param closing_func closing function
	 */
	template<typename string_t=std::string, typename int_t=int>
	MapGPU(func_t func, int_t pardegree, string_t name,
	       closing_func_t closing_func,
	       int_t max_buffered_tuples=default_max_buffered_tuples,
	       int_t gpu_blocks=default_gpu_blocks,
	       int_t gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {false}
	{
		check_constructor_parameters(pardegree, max_buffered_tuples,
					     gpu_blocks, gpu_threads_per_block);
		std::vector<ff_node *> workers;
		for (int_t i = 0; i < pardegree; i++) {
			auto seq = new MapGPU_Node {func, name,
						    RuntimeContext {pardegree, i},
						    max_buffered_tuples,
						    gpu_blocks,
						    gpu_threads_per_block,
						    closing_func};
			workers.push_back(seq);
		}
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>
					 {pardegree});
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the
		// emitter, workers and collector
		ff::ff_farm::cleanup_all();

		std::cout << is_invocable<func_t, tuple_t &>::value << std::endl;
		std::cout << is_invocable<func_t, const tuple_t &, result_t &>::value
			  << std::endl;
		std::cout << is_invocable<func_t, tuple_t &, char *,
					  std::size_t>::value
			  << std::endl;
		std::cout << is_invocable<func_t, const tuple_t &, result_t &,
					  char *, std::size_t>::value
			  << std::endl;
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param func function to be executed on each input tuple
	 *  \param pardegree parallelism degree of the MapGPU operator
	 *  \param name string with the unique name of the MapGPU operator
	 *  \param max_buffered_tuples numbers of tuples to buffer on the GPU
	 *  \param gpu_blocks the number of blocks to use when calling the GPU kernel
	 *  \param gpu_threads_per_block number of GPU threads per block
	 *  \param closing_func closing function
	 *  \param routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template<typename string_t=std::string, typename int_t=int>
	MapGPU(func_t func, int_t pardegree, string_t name,
	       closing_func_t closing_func, routing_func_t routing_func,
	       int_t max_buffered_tuples=default_max_buffered_tuples,
	       int_t gpu_blocks=default_gpu_blocks,
	       int_t gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {true}
	{
		check_constructor_parameters(pardegree, max_buffered_tuples,
					     gpu_blocks, gpu_threads_per_block);
		if (cudaMalloc(&scratchpads,
			       NUMBER_OF_KEYS * SCRATCHPAD_SIZE) != cudaSuccess)
			failwith("Failed to allocate scratchpad area");
		std::vector<ff_node *> workers;
		for (int_t i = 0; i < pardegree; i++) {
			auto seq = new MapGPU_Node {func, name,
						    RuntimeContext {pardegree, i},
						    max_buffered_tuples,
						    gpu_blocks,
						    gpu_threads_per_block,
						    scratchpads,
						    NUMBER_OF_KEYS,
						    SCRATCHPAD_SIZE,
						    closing_func};
			workers.push_back(seq);
		}
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>
					 {routing_func, pardegree});
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the
		// emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	~MapGPU() { cudaFree(scratchpads); }

	/**
	 *  \brief Check whether the MapGPU has been instantiated with a key-based distribution or not
	 *  \return true if the MapGPU is configured with keyBy
	 */
	bool isKeyed() const { return is_keyed; }

	/**
	 *  \brief Check whether the Map has been used in a MultiPipe
	 *  \return true if the Map has been added/chained to an existing MultiPipe
	 */
	bool isUsed() const { return used; }

	/// deleted constructors/operators. This object may not be copied nor
	/// moved.
	MapGPU(const MapGPU &) = delete;
	MapGPU(MapGPU &&) = delete;
	MapGPU &operator=(const MapGPU &) = delete;
	MapGPU &operator=(MapGPU &&) = delete;
};
} // namespace wf

#endif
