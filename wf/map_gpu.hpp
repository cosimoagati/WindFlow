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

namespace wf {
inline void check_constructor_parameters(int pardegree,
					 int tuple_buffer_capacity,
					 int gpu_threads_per_block) {
	if (pardegree <= 0) {
		failwith("MapGPU has non-positive parallelism");
	}
	if (tuple_buffer_capacity <= 0) {
		failwith("MapGPU has non-positive maximum buffered tuples");
	}
	if (gpu_threads_per_block <= 0) {
		failwith("MapGPU has non-positive number of "
			 "GPU threads per block");
	}
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
class MapGPU: public ff::ff_farm {
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

	/*
	 * Make sure the function to be computed has a valid signature.
	 */
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
	static_assert(!(is_in_place_keyless<func_t> || is_in_place_keyed<func_t>)
		      || std::is_same<tuple_t, result_t>::value,
		      "WindFlow Error: if instantiating MapGPU with an in-place "
		      "function, the input type and the output type must match.");

	/*
	 * Type aliases.
	 */
	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t,
							 std::size_t)>;

	// The type of objects used as internal farm nodes.
	using node_t = MapGPU_Node<tuple_t, result_t, func_t>;

	static constexpr auto default_tuple_buffer_capacity = 256;
	static constexpr auto default_gpu_threads_per_block = 256;
	static constexpr auto default_scratchpad_size = 64;

	bool is_used {false}; // is the MapGPU used in a MultiPipe or not?
	bool is_keyed; // is the MapGPU is configured with keyBy or not?

	friend class MultiPipe;
public:
	/**
	 *  \brief Constructor I
	 *
	 *  \param func function to be executed on each input tuple
	 *  \param pardegree parallelism degree of the MapGPU operator
	 *  \param name string with the unique name of the MapGPU operator
	 *  \param tuple_buffer_capacity numbers of tuples to buffer on the GPU
	 *  \param gpu_threads_per_block number of GPU threads per block
	 */
	// TODO: It would be nice to factor out common constructor behavior...
	MapGPU(func_t func, int pardegree, std::string name,
	       int tuple_buffer_capacity=default_tuple_buffer_capacity,
	       int gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {false}
	{
		check_constructor_parameters(pardegree, tuple_buffer_capacity,
					     gpu_threads_per_block);
		std::vector<ff_node *> workers;
		for (auto i = 0; i < pardegree; i++) {
			auto seq = new node_t {func, name,
					       tuple_buffer_capacity,
					       gpu_threads_per_block};
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
	 *  \param gpu_threads_per_block number of GPU threads per block
	 *  \param routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	MapGPU(func_t func, int pardegree, std::string name,
	       routing_func_t routing_func,
	       int tuple_buffer_capacity=default_tuple_buffer_capacity,
	       int gpu_threads_per_block=default_gpu_threads_per_block,
	       int scratchpad_size=default_scratchpad_size)
		: is_keyed {true}
	{
		check_constructor_parameters(pardegree, tuple_buffer_capacity,
					     gpu_threads_per_block);
		if (scratchpad_size <= 0) {
			failwith("MapGPU has non-positive scratchpad size");
		}
		std::vector<ff_node *> workers;
		for (auto i = 0; i < pardegree; i++) {
			auto seq = new node_t {func, name,
					       tuple_buffer_capacity,
					       gpu_threads_per_block,
					       scratchpad_size};
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

	/*
	 * This object may not be copied nor moved.
	 */
	MapGPU(const MapGPU &) = delete;
	MapGPU(MapGPU &&) = delete;
	MapGPU &operator=(const MapGPU &) = delete;
	MapGPU &operator=(MapGPU &&) = delete;
};
} // namespace wf

#endif
