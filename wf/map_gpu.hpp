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
 *  @brief MapGPU operator executing a one-to-one transformation on the input
 *  stream
 *
 *  @section Map_GPU (Description)
 *
 *  This file implements the Map operator able to execute a one-to-one
 *  transformation on each tuple of the input data stream. The transformation
 *  should be stateless and must produce one output result for each input tuple
 *  consumed (in the keyless version) or stateful.
 *
 *  The template parameters tuple_t and result_t must be default constructible,
 *  with a copy Constructor and copy assignment operator, and they must provide
 *  and implement the setControlFields() and getControlFields() methods.
 */

#ifndef MAP_GPU_H
#define MAP_GPU_H

#include <cstdlib>
#include <string>
#include <vector>

#include <ff/farm.hpp>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp"

#include "map_gpu_node.hpp"
#include "map_gpu_utils.hpp"

namespace wf
{
inline void
check_constructor_parameters(int pardegree, int tuple_buffer_capacity,
			     int gpu_blocks, int gpu_threads_per_block)
{
	if (pardegree <= 0)
		failwith("MapGPU has non-positive parallelism");
	if (tuple_buffer_capacity <= 0)
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
// TODO: Can we put set the result_t parameter to be the same as tuple_t
// if not specified?
template<typename tuple_t, typename result_t, typename func_t>
class MapGPU: public ff::ff_farm
{
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

	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t,
							 std::size_t)>;
	using node_t = MapGPU_Node<tuple_t, result_t, func_t,
				   closing_func_t, routing_func_t>;

	static constexpr auto default_tuple_buffer_capacity = 256;
	static constexpr auto default_gpu_threads_per_block = 256;

	// Arbitrary default values for now.
	static constexpr auto NUMBER_OF_KEYS = 256;
	static constexpr auto SCRATCHPAD_SIZE = 64; // Size of a single
						    // scratchpad in chars.
	char *scratchpads {nullptr}; // Points to the GPU memory area containing
				     // the scratchpads for stateful keyed
				     // functions.
	bool is_keyed; // is the MapGPU is configured with keyBy or not?
	bool is_used; // is the MapGPU used in a MultiPipe or not?

	// friendships with other classes in the library
	friend class MultiPipe;
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
	 *  \param tuple_buffer_capacity numbers of tuples to buffer on the GPU
	 *  \param gpu_blocks the number of blocks to use when calling the GPU kernel
	 *  \param gpu_threads_per_block number of GPU threads per block
	 *  \param closing_func closing function
	 */
	// TODO: It would be nice to factor out common constructor behavior...
	template<typename string_t=std::string, typename int_t=int>
	MapGPU(func_t func, int_t pardegree, string_t name,
	       closing_func_t closing_func,
	       int_t tuple_buffer_capacity=default_tuple_buffer_capacity,
	       int_t gpu_blocks=default_gpu_blocks,
	       int_t gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {false}
	{
		check_constructor_parameters(pardegree, tuple_buffer_capacity,
					     gpu_blocks, gpu_threads_per_block);
		std::vector<ff_node *> workers;
		for (int_t i = 0; i < pardegree; i++) {
			auto seq = new node_t {func, name,
					       RuntimeContext {pardegree, i},
					       tuple_buffer_capacity,
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
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param func function to be executed on each input tuple
	 *  \param pardegree parallelism degree of the MapGPU operator
	 *  \param name string with the unique name of the MapGPU operator
	 *  \param tuple_buffer_capacity numbers of tuples to buffer on the GPU
	 *  \param gpu_blocks the number of blocks to use when calling the GPU kernel
	 *  \param gpu_threads_per_block number of GPU threads per block
	 *  \param closing_func closing function
	 *  \param routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template<typename string_t=std::string, typename int_t=int>
	MapGPU(func_t func, int_t pardegree, string_t name,
	       closing_func_t closing_func, routing_func_t routing_func,
	       int_t tuple_buffer_capacity=default_tuple_buffer_capacity,
	       int_t gpu_blocks=default_gpu_blocks,
	       int_t gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {true}
	{
		check_constructor_parameters(pardegree, tuple_buffer_capacity,
					     gpu_blocks, gpu_threads_per_block);
		if (cudaMalloc(&scratchpads,
			       NUMBER_OF_KEYS * SCRATCHPAD_SIZE) != cudaSuccess)
			failwith("Failed to allocate scratchpad area");
		std::vector<ff_node *> workers;
		for (int_t i = 0; i < pardegree; i++) {
			auto seq = new node_t {func, name,
					       RuntimeContext {pardegree, i},
					       tuple_buffer_capacity,
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
	bool isUsed() const { return is_used; }

	/// deleted constructors/operators. This object may not be copied nor
	/// moved.
	MapGPU(const MapGPU &) = delete;
	MapGPU(MapGPU &&) = delete;
	MapGPU &operator=(const MapGPU &) = delete;
	MapGPU &operator=(MapGPU &&) = delete;
};
} // namespace wf

#endif
