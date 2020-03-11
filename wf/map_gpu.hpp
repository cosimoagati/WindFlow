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

/// includes
#include <cstdlib>
#include <string>
#include <vector>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include "basic.hpp"
#include "context.hpp"
#include "standard_nodes.hpp"

namespace wf {
/**
 *  \class Map
 *
 *  \brief Map operator executing a one-to-one transformation on the input stream
 *
 *  This class implements the Map operator executing a one-to-one stateless transformation
 *  on each tuple of the input stream.
 */
template<typename tuple_t, typename result_t>
class MapGPU: public ff::ff_farm
{
public:
	/// type of the map function (in-place version)
	using map_func_ip_t = std::function<void(tuple_t &)>;
	/// type of the map function (not in-place version)
	using map_func_nip_t = std::function<void(const tuple_t &, result_t &)>;
	/// type of the closing function
	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t,
							 std::size_t)>;
private:
	// friendships with other classes in the library
	friend class MultiPipe;
	bool keyed; // is the MapGPU is configured with keyBy or not?
	// bool used; // (Doesn't seem to be used for now).

	class MapGPU_Node: public ff::ff_node_t<tuple_t, result_t>
	{
		static constexpr auto max_buffered_tuples = 256;
		tuple_t *tuple_buffer;
		result_t *result_buffer;

		map_func_ip_t func_ip; // in-place map function
		map_func_nip_t func_nip; // not in-place map function
		closing_func_t closing_func; // closing function
		std::string name; // string of the unique name of the operator
		bool isIP; // flag stating if the in-place map function should
			   // be used (otherwise the not in-place version)
		RuntimeContext context; // RuntimeContext
		decltype(max_buffered_tuples) buf_index {0};

#if defined(TRACE_WINDFLOW)
		unsigned long rcvTuples {0};
		double avg_td_us {0};
		double avg_ts_us {0};
		volatile unsigned long startTD, startTS, endTD, endTS;
		std::ofstream *logfile = nullptr;
#endif
		__global__ void map_kernel_ip(const tuple_t *buffer,
					      const std::size_t buffer_size,
					      const map_func_ip_t f)
		{
			const auto index = blockIdx.x * blockDim.x + threadIdx.x;
			const auto stride = blockDim.x * gridDim.x;
			for (auto i = index; i < buffer_size; i += stride)
				f(buffer[i]);
		}
		__global__ void map_kernel_nip(const tuple_t *input_buffer,
					       const result_t *result_buffer,
					       const std::size_t buffer_size,
					       const map_func_nip_t f)
		{
			const auto index = blockIdx.x * blockDim.x + threadIdx.x;
			const auto stride = blockDim.x * gridDim.x;
			for (auto i = index; i < buffer_size; i += stride)
				result_buffer[i] = f(input_buffer[i]);
		}

		inline void fill_tuple_buffer(tuple_t *t)
		{
			tuple_buffer[buf_index] = *t;
			buf_index++;
			delete t;
		}

		inline void send_mapped_tuples()
		{
			const auto &output_buffer = isIP
				? tuple_buffer
				: result_buffer;
			for (auto i = 0; i < max_buffered_tuples; ++i)
				ff_send_out(new result_t
					    {reinterpret_cast<result_t>(output_buffer[i])});
			buf_index = 0;
		}
	public:
		// Constructor I
		template <typename T=std::string>
		MapGPU_Node(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
			    T _name,
			    RuntimeContext _context,
			    closing_func_t _closing_func):
			func_ip(_func),
			name(_name),
			isIP(true),
			context(_context),
			closing_func(_closing_func)
		{}

		// Constructor III
		template <typename T=std::string>
		MapGPU_Node(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
			    T _name,
			    RuntimeContext _context,
			    closing_func_t _closing_func):
			func_nip(_func),
			name(_name),
			isIP(false),
			context(_context),
			closing_func(_closing_func)
		{}

		// svc_init method (utilized by the FastFlow runtime)
		int svc_init()
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
			cudaMallocManaged(&tuple_buffer,
					  max_buffered_tuples * sizeof(tuple_t));
			cudaMallocManaged(&result_buffer,
					  max_buffered_tuples * sizeof(tuple_t));
			return 0;
		}

		// svc method (utilized by the FastFlow runtime)
		result_t *svc(tuple_t *t)
		{
#if defined (TRACE_WINDFLOW)
			startTS = current_time_nsecs();
			if (rcvTuples == 0)
				startTD = current_time_nsecs();
			rcvTuples++;
#endif
			// in-place version
			if (buf_index < max_buffered_tuples) {
				fill_tuple_buffer(t);
				return GO_ON;
			}
			if (isIP)
				map_kernel_ip<<<1, 32>>>(tuple_buffer,
							 max_buffered_tuples,
							 func_ip);
			else
				map_kernel_nip<<<1, 32>>>(tuple_buffer,
							  result_buffer,
							  max_buffered_tuples,
							  func_nip);
			cudaDeviceSynchronize();
			send_mapped_tuples();
#if defined(TRACE_WINDFLOW)
			endTS = current_time_nsecs();
			endTD = current_time_nsecs();
			double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
			avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
			double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
			avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
			startTD = current_time_nsecs();
#endif
			return GO_ON;
		}

		// svc_end method (utilized by the FastFlow runtime)
		void svc_end()
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
			cudaFree(tuple_buffer);
			cudaFree(result_buffer);
		}
	};

public:
	/**
	 *  \brief Constructor I
	 *
	 *  \param _func function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the MapGPU operator
	 *  \param _name string with the unique name of the MapGPU operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	MapGPU(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
	       T _pardegree,
	       std::string _name,
	       closing_func_t _closing_func):
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED
				  << "WindFlow Error: MapGPU has parallelism zero"
				  << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of MapGPU_Node
		std::vector<ff_node *> workers;
		for (std::size_t i = 0; i < _pardegree; i++) {
			auto *seq = new MapGPU_Node(_func, _name,
						    RuntimeContext(_pardegree,
								   i),
						    _closing_func);
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param _func function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the MapGPU operator
	 *  \param _name string with the unique name of the MapGPU operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template <typename T=std::size_t>
	MapGPU(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
	       T _pardegree,
	       std::string _name,
	       closing_func_t _closing_func,
	       routing_func_t _routing_func):
		keyed(true)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED
				  << "WindFlow Error: MapGPU has parallelism zero"
				  << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of MapGPU_Node
		std::vector<ff_node *> workers;
		for (std::size_t i = 0; i < _pardegree; i++) {
			auto *seq = new MapGPU_Node(_func, _name,
						    RuntimeContext(_pardegree,
								   i),
						    _closing_func);
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func,
								       _pardegree));
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor V
	 *
	 *  \param _func function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the MapGPU operator
	 *  \param _name string with the unique name of the MapGPU operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	MapGPU(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
	       T _pardegree,
	       std::string _name,
	       closing_func_t _closing_func):
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: MapGPU has parallelism zero" << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of MapGPU_Node
		std::vector<ff_node *> workers;
		for (std::size_t i = 0; i < _pardegree; i++) {
			auto *seq = new MapGPU_Node(_func, _name,
						    RuntimeContext(_pardegree,
								   i),
						    _closing_func);
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor VI
	 *
	 *  \param _func function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the MapGPU operator
	 *  \param _name string with the unique name of the MapGPU operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template <typename T=std::size_t>
	MapGPU(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
	       T _pardegree,
	       std::string _name,
	       closing_func_t _closing_func,
	       routing_func_t _routing_func):
		keyed(true)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED
				  << "WindFlow Error: MapGPU has parallelism zero"
				  << DEFAULT << std::endl;
			std::exit(EXIT_FAILURE);
		}
		// vector of MapGPU_Node
		std::vector<ff_node *> workers;
		for (std::size_t i = 0; i < _pardegree; i++) {
			auto *seq = new MapGPU_Node(_func, _name,
						    RuntimeContext(_pardegree, i),
						    _closing_func);
			workers.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
		// add workers
		ff::ff_farm::add_workers(workers);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the MapGPU will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Check whether the MapGPU has been instantiated with a key-based distribution or not
	 *  \return true if the MapGPU is configured with keyBy
	 */
	bool isKeyed() const
	{
		return keyed;
	}
};

} // namespace wf

#endif
