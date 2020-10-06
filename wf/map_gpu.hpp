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

#include "basic.hpp"
#include "basic_operator.hpp"
#include "context.hpp"
#include <ff/farm.hpp>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
// #include "standard_emitter.hpp"
#include "standard_nodes_gpu.hpp"

#include "gpu_utils.hpp"
#include "map_gpu_node.hpp"

namespace wf {
inline void check_constructor_parameters(const int pardegree, const int tuple_buffer_capacity,
                                         const int gpu_threads_per_block) {
	if (pardegree <= 0)
		failwith("MapGPU has non-positive parallelism");
	if (tuple_buffer_capacity <= 0)
		failwith("MapGPU has non-positive maximum buffered tuples");
	if (gpu_threads_per_block <= 0)
		failwith("MapGPU has non-positive number of GPU threads per block");
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
class MapGPU : public ff::ff_farm, public Basic_Operator {
	/*
	 * Name function properties, used to verify compile-time invariants and
	 * only compile the required member functions.  These predicates cannot
	 * be declared as auto, since they would be treated as redeclarations
	 * otherwise (without an explicit type, they would be considered
	 * incompatible with any previous instantiation, that's why using them
	 * just once with auto works, but not if used more than once).
	 */
	template<typename F> static constexpr bool is_in_place_keyless = is_invocable<F, tuple_t &>::value;

	template<typename F>
	static constexpr bool is_not_in_place_keyless = is_invocable<F, const tuple_t &, tuple_t &>::value;

	template<typename F>
	static constexpr bool is_in_place_keyed = is_invocable<F, tuple_t &, char *, std::size_t>::value;

	template<typename F>
	static constexpr bool is_not_in_place_keyed =
	        is_invocable<F, const tuple_t &, result_t &, char *, std::size_t>::value;

	/*
	 * Make sure the function to be computed has a valid signature.
	 */
	static_assert(
	        (is_in_place_keyless<
	                 func_t> && !is_not_in_place_keyless<func_t> && !is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
	                || (!is_in_place_keyless<
	                            func_t> && is_not_in_place_keyless<func_t> && !is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
	                || (!is_in_place_keyless<
	                            func_t> && !is_not_in_place_keyless<func_t> && is_in_place_keyed<func_t> && !is_not_in_place_keyed<func_t>)
	                || (!is_in_place_keyless<
	                            func_t> && !is_not_in_place_keyless<func_t> && !is_in_place_keyed<func_t> && is_not_in_place_keyed<func_t>),
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

	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t, std::size_t)>;

	// The type of objects used as internal farm nodes.
	using node_t = MapGPU_Node<tuple_t, result_t, func_t>;

	static constexpr auto default_tuple_buffer_capacity = 256;
	static constexpr auto default_gpu_threads_per_block = 256;
	static constexpr auto default_scratchpad_size       = 64;

	std::string     name;
	std::size_t     pardegree;
	routing_modes_t routing_mode;

	bool is_used {false}; // is the MapGPU used in a MultiPipe or not?
	bool have_gpu_input;  // is the MapGPU receiving input from another operator on the GPU?
	bool have_gpu_output; // is the MapGPU sending output to another operator on the GPU?

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
	MapGPU(func_t func, const int pardegree, const std::string &name,
	       const int  tuple_buffer_capacity = default_tuple_buffer_capacity,
	       const int  gpu_threads_per_block = default_gpu_threads_per_block,
	       const bool have_gpu_input = false, const bool have_gpu_output = false)
	        : name {name}, pardegree {pardegree}, routing_mode {FORWARD}, have_gpu_input {have_gpu_input},
	          have_gpu_output {have_gpu_output} {
		check_constructor_parameters(pardegree, tuple_buffer_capacity, gpu_threads_per_block);
		std::vector<ff_node *> workers;
		for (auto i = 0; i < pardegree; i++) {
			auto seq = new node_t {
			        func, name,           tuple_buffer_capacity, gpu_threads_per_block,
			        0,    have_gpu_input, have_gpu_output};
			workers.push_back(seq);
		}
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t> {pardegree});
		ff::ff_farm::add_emitter(new Standard_EmitterGPU<tuple_t> {pardegree, have_gpu_input});
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
	 *  \param routing_func function to map the key hashcode onto an identifier starting from zero to
	 * pardegree-1 \param tuple_buffer_capacity numbers of tuples to buffer on the GPU \param
	 * gpu_threads_per_block number of GPU threads per block \param scratchpad_size size of the scratchpad
	 * in case of keyed computations. \param have_gpu_input specifies if the operator is receiving input
	 * directly from the GPU \param have_gpu_output specifies if the operator is sending output directly
	 * to the GPU \param is_keyed specifies whether the operator is operating via key-based routing
	 */
	MapGPU(func_t func, const int pardegree, const std::string &name, routing_func_t routing_func,
	       const int tuple_buffer_capacity = default_tuple_buffer_capacity,
	       const int gpu_threads_per_block = default_gpu_threads_per_block,
	       const int scratchpad_size = default_scratchpad_size, const bool have_gpu_input = false,
	       const bool have_gpu_output = false)
	        : name {name}, pardegree {pardegree}, routing_mode {KEYBY}, have_gpu_input {have_gpu_input},
	          have_gpu_output {have_gpu_output} {
		check_constructor_parameters(pardegree, tuple_buffer_capacity, gpu_threads_per_block);
		if (scratchpad_size <= 0)
			failwith("MapGPU has non-positive scratchpad size");
		std::vector<ff_node *> workers;
		for (auto i = 0; i < pardegree; i++) {
			auto seq = new node_t {func,
			                       name,
			                       tuple_buffer_capacity,
			                       gpu_threads_per_block,
			                       scratchpad_size,
			                       have_gpu_input,
			                       have_gpu_output};
			workers.push_back(seq);
		}
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>
		// 			 {routing_func, pardegree});
		ff::ff_farm::add_emitter(
		        new Standard_EmitterGPU<tuple_t> {routing_func, pardegree, have_gpu_input});
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need also to destroy the
		// emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	void set_GPUInput(const bool val) {
		for (auto worker : this->getWorkers()) {
			auto worker_casted = static_cast<node_t *>(worker);
			worker_casted->set_GPUInput(val);
		}
	}

	void set_GPUOutput(const bool val) {
		for (auto worker : this->getWorkers()) {
			auto worker_casted = static_cast<node_t *>(worker);
			worker_casted->set_GPUOutput(val);
		}
	}

	/**
	 *  \brief Get the name of the operator
	 *  \return name of the operator
	 */
	std::string getName() const override { return name; }

	/**
	 *  \brief Get the total parallelism within the operator
	 *  \return total parallelism within the operator
	 */
	std::size_t getParallelism() const override { return pardegree; }

	/**
	 *  \brief Return the routing mode of the operator
	 *  \return routing mode used by the operator
	 */
	routing_modes_t getRoutingMode() const override { return routing_mode; }

	/**
	 *  \brief Check whether the MapGPU has been used in a MultiPipe
	 *  \return true if the MapGPU has been added/chained to an existing MultiPipe
	 */
	bool isUsed() const override { return is_used; }

	/**
	 *  \brief Get the Stats_Record of each replica within the operator
	 *  \return vector of Stats_Record objects
	 */
	std::vector<Stats_Record> get_StatsRecords() const override {
		// TODO
		return {Stats_Record {name, name, true}};
	}

	/**
	 *  \brief Check whether the MapGPU has input from another GPU operator
	 *  \return true if the MapGPU has input on the GPU, false otherwise
	 */
	bool HasGPUInput() const { return have_gpu_input; }

	/**
	 *  \brief Check whether the MapGPU has output to another GPU operator
	 *  \return true if the MapGPU has output to the GPU, false otherwise
	 */
	bool HasGPUOutput() const { return have_gpu_output; }
};
} // namespace wf

#endif
