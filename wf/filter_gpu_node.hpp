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

#include <ff/node.hpp>

#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp" // Probably not required...

namespace wf
{
template<typename tuple_t, typename func_t, typename closing_func_t,
	 typename fil>
class FilterGPU_Node: public ff::ff_node_t<tuple_t>
{
	static constexpr auto tuple_buffer_capacity = 256;

	tuple_t *cpu_tuple_buffer;
	tuple_t *gpu_tuple_buffer;
	bool *cpu_tuple_mask;
	bool *gpu_tuple_mask;

	func_t filter_func; // filter function (predicate)
	closing_func_t closing_func; // closing function
	std::string name; // string of the unique name of the operator
	RuntimeContext context;
	std::size_t buf_index {0};

#if defined(TRACE_WINDFLOW)
	unsigned long rcvTuples {0};
	double avg_td_us {0};
	double avg_ts_us {0};
	volatile unsigned long startTD, startTS, endTD, endTS;
	std::ofstream *logfile = nullptr;
#endif
public:
	// Constructor I
	FilterGPU_Node(func_t _filter_func, std::string _name,
		       RuntimeContext _context, closing_func_t _closing_func):
		filter_func(_filter_func), name(_name),
		context(_context), closing_func(_closing_func)
	{}

	// svc_init method (utilized by the FastFlow runtime)
	int
	svc_init()
	{
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
		cudaMallocManaged(tuple_buffer,
				  max_buffered_tuples * sizeof(tuple_t));
		cudaMallocManaged(&tuple_mask_array,
				  max_buffered_tuples * sizeof(bool));
		return 0;
	}

	// svc method (utilized by the FastFlow runtime)
	tuple_t *
	svc(tuple_t *t)
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
		filter_kernel<tuple_t, filter_func_t>
			<<<1, 32>>>(gpu_tuple_buffer, tuple_mask_array,
				    max_buffered_tuples, filter_func);
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
	void
	svc_end()
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
		cudaFree(cpu_tuple_buffer);
		cudaFree(cpu_tuple_mask);
	}
};
}

#endif
