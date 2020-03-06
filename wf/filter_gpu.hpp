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
 *  @brief FilterGPU operator dropping data items not respecting a given predicate
 *
 *  @section FilterGPU (Description)
 *
 *  This file implements the FilterGPU operator able to drop all the input items that do not
 *  respect a given predicate given by the user.
 *
 *  The template parameter tuple_t must be default constructible, with a copy constructor
 *  and copy assignment operator, and it must provide and implement the setControlFields()
 *  and getControlFields() methods.
 */

#ifndef FILTER_GPU_H
#define FILTER_GPU_H

/// includes
#include <cstdlib>
#include <array>
#include <string>
#include <iostream>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp"

namespace wf {
/**
 *  \class FilterGPU
 *
 *  \brief FilterGPU operator dropping data items not respecting a given predicate
 *
 *  This class implements the FilterGPU operator applying a given predicate to all the input
 *  items and dropping out all of them for which the predicate evaluates to false.
 */
template<typename tuple_t>
class FilterGPU: public ff::ff_farm
{
public:
	/// type of the predicate function
	using filter_func_t = std::function<bool(tuple_t &)>;
	/// type of the rich predicate function
	using rich_filter_func_t = std::function<bool(tuple_t &, RuntimeContext &)>;
	/// type of the closing function
	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;

private:
	// friendships with other classes in the library
	friend class MultiPipe;
	bool used; // true if the operator has been added/chained in a MultiPipe
	bool keyed; // flag stating whether the FilterGPU is configured with keyBy or not
	// class FilterGPU_Node
	class FilterGPU_Node: public ff::ff_node_t<tuple_t>
	{
		static constexpr auto max_buffered_tuples = 256;
		std::array<tuple_t *, max_buffered_tuples> cpu_tuple_buffer;
		tuple_t *gpu_tuple_buffer;
		bool *tuple_mask_array;

		filter_func_t filter_func; // filter function (predicate)
		rich_filter_func_t rich_filter_func; // rich filter function (predicate)
		closing_func_t closing_func; // closing function
		std::string name; // string of the unique name of the operator
		bool isRich; // flag stating whether the function to be used is rich (i.e. it receives the RuntimeContext object)
		RuntimeContext context; // RuntimeContext
		decltype(max_buffered_tuples) buf_index {0};

#if defined(TRACE_WINDFLOW)
		unsigned long rcvTuples {0};
		double avg_td_us {0};
		double avg_ts_us {0};
		volatile unsigned long startTD, startTS, endTD, endTS;
		std::ofstream *logfile = nullptr;
#endif
		/**
		 * Individual results are stored in a boolean mask array, which the CPU
		 * then uses to determine which tuples to send out.
		 */
		__global__ void filter_kernel(const tuple_t *tuple_buffer,
					      const bool *tuple_mask_array,
					      const std::size_t buffer_size,
					      const filter_func_t f)
		{
			const auto index = blockIdx.x * blockDim.x + threadIdx.x;
			const auto stride = blockDim.x * gridDim.x;
			for (auto i = index; i < buffer_size; i += stride)
				tuple_mask_array[i] = f(tuple_buffer[i]);
		}
		inline void fill_tuple_buffers(tuple_t *t)
		{
			cpu_tuple_buffer[buf_index] = t;
			gpu_tuple_buffer[buf_index] = *t;
			buf_index++;
		}
		inline void send_filtered_tuples()
		{
			for (auto i = 0; i < max_buffered_tuples; ++i) {
				if (tuple_mask_array[i])
					ff_send_out(cpu_tuple_buffer[i]);
				else
					delete cpu_tuple_buffer[i];
			}
			buf_index = 0;
		}
	public:
		// Constructor I
		FilterGPU_Node(filter_func_t _filter_func,
			       std::string _name,
			       RuntimeContext _context,
			       closing_func_t _closing_func):
			filter_func(_filter_func),
			name(_name),
			isRich(false),
			context(_context),
			closing_func(_closing_func)
		{}

		// Constructor II
		FilterGPU_Node(rich_filter_func_t _rich_filter_func,
			       std::string _name,
			       RuntimeContext _context,
			       closing_func_t _closing_func):
			rich_filter_func(_rich_filter_func),
			name(_name), isRich(true),
			context(_context),
			closing_func(_closing_func)
		{}

		// svc_init method (utilized by the FastFlow runtime)
		int svc_init()
		{
#if defined(TRACE_WINDFLOW)
			logfile = new std::ofstream();
			name += "_" + std::to_string(this->get_my_id()) + "_" + std::to_string(getpid()) + ".log";
#if defined(LOG_DIR)
			std::string filename = std::string(STRINGIFY(LOG_DIR)) + "/" + name;
			std::string log_dir = std::string(STRINGIFY(LOG_DIR));
#else
			std::string filename = "log/" + name;
			std::string log_dir = std::string("log");
#endif
			// create the log directory
			if (mkdir(log_dir.c_str(), 0777) != 0) {
				struct stat st;
				if((stat(log_dir.c_str(), &st) != 0) || !S_ISDIR(st.st_mode)) {
					std::cerr << RED << "WindFlow Error: directory for log files cannot be created" << DEFAULT_COLOR << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
			logfile->open(filename);

#endif
			cudaMallocManaged(&tuple_buffer,
					  max_buffered_tuples * sizeof(tuple_t));
			cudaMallocManaged(&tuple_mask_array,
					  max_buffered_tuples * sizeof(bool));
			return 0;
		}

		// svc method (utilized by the FastFlow runtime)
		tuple_t *svc(tuple_t *t)
		{
#if defined(TRACE_WINDFLOW)
			startTS = current_time_nsecs();
			if (rcvTuples == 0)
				startTD = current_time_nsecs();
			rcvTuples++;
#endif
			if (buf_index < max_buffered_tuples) {
				fill_tuple_buffers(t);
				return GO_ON;
			}
			// evaluate the predicate on buffered items
			filter_kernel<<<1, 32>>>(gpu_tuple_buffer,
						 tuple_mask_array,
						 max_buffered_tuples,
						 filter_func);
			cudaDeviceSynchronize();
#if defined(TRACE_WINDFLOW)
			endTS = current_time_nsecs();
			endTD = current_time_nsecs();
			double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
			avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
			double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
			avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
			startTD = current_time_nsecs();
#endif
			send_filtered_tuples();
			return GO_ON;
		}

		// svc_end method (utilized by the FastFlow runtime)
		void svc_end()
		{
			// call the closing function
			closing_func(context);
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
			cudaFree(tuple_buffer);
			cudaFree(tuple_mask_array);
		}
	};
public:
	/**
	 *  \brief Constructor I
	 *
	 *  \param _func filter function (boolean predicate)
	 *  \param _pardegree parallelism degree of the FilterGPU operator
	 *  \param _name string with the unique name of the FilterGPU operator
	 *  \param _closing_func closing function
	 */
	FilterGPU(filter_func_t _func,
		  size_t _pardegree,
		  std::string _name,
		  closing_func_t _closing_func):
		keyed(false), used(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: FilterGPU has parallelism zero" << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> w;
		for (size_t i=0; i<_pardegree; i++) {
			auto *seq = new FilterGPU_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
			w.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		ff::ff_farm::add_workers(w);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param _func filter function (boolean predicate)
	 *  \param _pardegree parallelism degree of the FilterGPU operator
	 *  \param _name string with the unique name of the FilterGPU operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	FilterGPU(filter_func_t _func,
		  size_t _pardegree,
		  std::string _name,
		  closing_func_t _closing_func,
		  routing_func_t _routing_func):
		keyed(true), used(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: FilterGPU has parallelism zero" << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> w;
		for (size_t i=0; i<_pardegree; i++) {
			auto *seq = new FilterGPU_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
			w.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
		// add workers
		ff::ff_farm::add_workers(w);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor III
	 *
	 *  \param _func rich filter function (boolean predicate)
	 *  \param _pardegree parallelism degree of the FilterGPU operator
	 *  \param _name string with the unique name of the FilterGPU operator
	 *  \param _closing_func closing function
	 */
	FilterGPU(rich_filter_func_t _func,
		  size_t _pardegree,
		  std::string _name,
		  closing_func_t _closing_func):
		keyed(false), used(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: FilterGPU has parallelism zero" << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> w;
		for (size_t i=0; i<_pardegree; i++) {
			auto *seq = new FilterGPU_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
			w.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		ff::ff_farm::add_workers(w);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor IV
	 *
	 *  \param _func rich filter function (boolean predicate)
	 *  \param _pardegree parallelism degree of the FilterGPU operator
	 *  \param _name string with the unique name of the FilterGPU operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	FilterGPU(rich_filter_func_t _func,
		  size_t _pardegree,
		  std::string _name,
		  closing_func_t _closing_func,
		  routing_func_t _routing_func):
		keyed(true), used(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: FilterGPU has parallelism zero" << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of FilterGPU_Node
		std::vector<ff_node *> w;
		for (size_t i=0; i<_pardegree; i++) {
			auto *seq = new FilterGPU_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
			w.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
		// add workers
		ff::ff_farm::add_workers(w);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the FilterGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Check whether the FilterGPU has been instantiated with a key-based distribution or not
	 *  \return true if the FilterGPU is configured with keyBy
	 */
	bool isKeyed() const
	{
		return keyed;
	}

	/**
	 *  \brief Check whether the Filter has been used in a MultiPipe
	 *  \return true if the Filter has been added/chained to an existing MultiPipe
	 */
	bool isUsed() const
	{
		return used;
	}

	/// deleted constructors/operators
	FilterGPU(const FilterGPU &) = delete; // copy constructor
	FilterGPU(FilterGPU &&) = delete; // move constructor
	FilterGPU &operator=(const FilterGPU &) = delete; // copy assignment operator
	FilterGPU &operator=(FilterGPU &&) = delete; // move assignment operator
};

} // namespace wf

#endif
