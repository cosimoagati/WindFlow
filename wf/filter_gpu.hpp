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
 *  @file    filter.hpp
 *  @author  Cosimo Agati
 *  @date    10/01/2019
 *
 *  @brief FilterGPU operator dropping data items not respecting a given
 *  predicate
 *
 *  @section FilterGPU (Description)
 *
 *  This file implements the FilterGPU operator able to drop all the input items
 *  that do not respect a given predicate given by the user.
 *
 *  The template parameter tuple_t must be default constructible, with a copy
 *  constructor and copy assignment operator, and it must provide and implement
 *  the setControlFields() and getControlFields() methods.
 */

#ifndef FILTER_GPU_H
#define FILTER_GPU_H

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp"
#include "filter_gpu_node.hpp"

namespace wf {
/**
 *  \class FilterGPU
 *
 *  \brief FilterGPU operator dropping data items not respecting a given predicate
 *
 *  This class implements the FilterGPU operator applying a given predicate to all the input
 *  items and dropping out all of them for which the predicate evaluates to false.
 */
template<typename tuple_t, typename func_t>
class FilterGPU: public ff::ff_farm {
	using routing_func_t = std::function<std::size_t(std::size_t,
							 std::size_t)>;
	using node_t = FilterGPU_Node<tuple_t, func_t>;

	friend class MultiPipe;

	static constexpr auto default_tuple_buffer_capacity = 256;
	static constexpr auto default_gpu_threads_per_block = 256;
	static constexpr auto default_scratchpad_size = 64;
	bool is_used {false}; // Is this FilterGPU used in a MultiPipe?
	bool is_keyed; // Is this FilterGPU keyed?

public:
	/**
	 *  \brief Constructor I
	 *
	 *  \param func filter function (boolean predicate)
	 *  \param pardegree parallelism degree of the FilterGPU operator
	 *  \param name string with the unique name of the FilterGPU operator
	 *  \param tuple_buffer_capacity numbers of tuples to buffer on the GPU
	 *  \param gpu_threads_per_block number of GPU threads per block
	 */
	FilterGPU(func_t func, const int pardegree, const std::string name,
		  const int tuple_buffer_capacity=default_tuple_buffer_capacity,
		  const int gpu_threads_per_block=default_gpu_threads_per_block)
		: is_keyed {false}
	{
		if (pardegree == 0) {
			std::cerr << RED
				  << "WindFlow Error: FilterGPU has parallelism zero"
				  << DEFAULT_COLOR << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> workers;
		for (auto i = 0; i < pardegree; ++i) {
			auto seq = new node_t {func, name,
					       tuple_buffer_capacity,
					       gpu_threads_per_block);
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t> {pardegree});
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param func filter function (boolean predicate)
	 *  \param pardegree parallelism degree of the FilterGPU operator
	 *  \param name string with the unique name of the FilterGPU operator
	 *  \param routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 *  \param tuple_buffer_capacity numbers of tuples to buffer on the GPU
	 *  \param gpu_threads_per_block number of GPU threads per block
	 */
	FilterGPU(func_t func, int pardegree, std::string name,
		  routing_func_t routing_func,
		  const int tuple_buffer_capacity=default_tuple_buffer_capacity,
		  const int gpu_threads_per_block=default_gpu_threads_per_block,
		  const int scratchpad_size=default_scratchpad_size)
		: keyed(true)
	{
		// check the validity of the parallelism degree
		if (pardegree == 0) {
			std::cerr << RED
				  << "WindFlow Error: FilterGPU has parallelism zero"
				  << DEFAULT_COLOR << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> workers;
		for (size_t i = 0; i < pardegree; i++) {
			auto seq = new node_t {func, name, closing_func};
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t> {routing_func, pardegree});
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Check whether the FilterGPU has been instantiated with a
	 *  key-based distribution or not
	 * \return true if the FilterGPU is configured with keyBy
	 */
	bool isKeyed() const { return is_keyed; }

	/**
	 *  \brief Check whether the Filter has been used in a MultiPipe
	 *  \return true if the Filter has been added/chained to an existing
	 *  MultiPipe
	 */
	bool isUsed() const { return is_used; }

	/*
	 * This object may not be copied nor moved.
	 */
	FilterGPU(const FilterGPU &) = delete; // copy constructor
	FilterGPU(FilterGPU &&) = delete; // move constructor
	FilterGPU &operator=(const FilterGPU &) = delete; // copy assignment operator
	FilterGPU &operator=(FilterGPU &&) = delete; // move assignment operator
};

} // namespace wf

#endif
