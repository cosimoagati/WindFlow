/**
 * Totally NOT hacking on the existing WindFlow implementation!
 */
#ifndef WINDFLOW_MAP_GPU_H
#define WINDFLOW_MAP_GPU_H

#include <cstdint>
#include <string>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <wf/basic.hpp>
#include <wf/context.hpp>
#include <wf/standard_nodes.hpp>

// This will be customizable to suit the particular application.
constexpr std::uint32_t MAX_BUFFERED_TUPLES = 1024;


// This was from win_seq_gpu.hpp, why did I inlcude it here?
// template<typename win_F_t>
// __global__ void kernelBatch(void *input_data,
//                             std::size_t *start,
//                             std::size_t *end,
//                             std::uint64_t *gwids,
//                             void *results,
//                             win_F_t F,
//                             std::size_t batch_len,
//                             char *scratchpad_memory,
//                             std::size_t scratchpad_size)
// {
// 	using input_t = decltype(get_tuple_t(F));
// 	using output_t = decltype(get_result_t(F));
// 	int id = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (id < batch_len) {
// 		const auto scratchpad_addr = scratchpad_size > 0 ?
// 			&scratchpad_memory[id * scratchpad_size] :
// 			nullptr;
// 		F(gwids[id], ((input_t *) input_data) + start[id],
// 		  &((output_t *) results)[id], end[id] - start[id],
// 		  scratchpad_addr);
// 	}
// }

namespace wf {
/**
 *  \class Map
 *
 *  \brief Map operator executing a one-to-one transformation on the input stream
 *
 *  This class implements the Map operator executing a one-to-one stateless
 *  transformation on each tuple of the input stream.
 *
 * For simplicity, the following abbreviations in variables are used:
 * ip:  in-place
 * nip: not in-place
 */
template<typename tuple_t, typename result_t>
class Map: public ff::ff_farm
{
public:
	/// type of the map function (in-place version)
	using map_func_ip_t = std::function<void(tuple_t &)>;
	/// type of the rich map function (in-place version)
	using rich_map_func_ip_t = std::function<void(tuple_t &,
						      RuntimeContext &)>;
	/// type of the map function (not in-place version)
	using map_func_nip_t = std::function<void(const tuple_t &, result_t &)>;
	/// type of the rich map function (not in-place version)
	using rich_map_func_nip_t = std::function<void(const tuple_t &,
						       result_t &,
						       RuntimeContext &)>;
	/// type of the closing function
	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier
	/// starting from zero to pardegree-1
	// using routing_func_t = std::function<std::size_t(std::size_t, std::size_t)>;

private:
	// friendships with other classes in the library
	friend class MultiPipe;
	bool keyed; // flag stating whether the Map is configured with keyBy or not
	// class Map_Node
	// class Map_Node: public ff::ff_node_t<tuple_t, result_t>
// 	{
// 	private:
// 		map_func_ip_t func_ip; // in-place map function
// 		rich_map_func_ip_t rich_func_ip; // in-place rich map function
// 		map_func_nip_t func_nip; // not in-place map function
// 		rich_map_func_nip_t rich_func_nip; // not in-place rich map function
// 		closing_func_t closing_func; // closing function
// 		std::string name; // string of the unique name of the operator
// 		bool isIP; // flag stating if the in-place map function should be used (otherwise the not in-place version)
// 		bool isRich; // flag stating whether the function to be used is rich (i.e. it receives the RuntimeContext object)
// 		RuntimeContext context; // RuntimeContext
// #if defined(LOG_DIR)
// 		unsigned long rcvTuples = 0;
// 		double avg_td_us = 0;
// 		double avg_ts_us = 0;
// 		volatile unsigned long startTD, startTS, endTD, endTS;
// 		std::ofstream *logfile = nullptr;
// #endif

// 	public:
// 		// Constructor I
// 		template <typename T=std::string>
// 		Map_Node(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value,
// 			 map_func_ip_t>::type _func,
// 			 T _name,
// 			 RuntimeContext _context,
// 			 closing_func_t _closing_func):
// 			func_ip(_func),
// 			name(_name),
// 			isIP(true),
// 			isRich(false),
// 			context(_context),
// 			closing_func(_closing_func)
// 		{}

// 		// Constructor II
// 		template <typename T=std::string>
// 		Map_Node(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value,
// 			 rich_map_func_ip_t>::type _func,
// 			 T _name,
// 			 RuntimeContext _context,
// 			 closing_func_t _closing_func):
// 			rich_func_ip(_func),
// 			name(_name),
// 			isIP(true),
// 			isRich(true),
// 			context(_context),
// 			closing_func(_closing_func)
// 		{}

// 		// Constructor III
// 		template <typename T=std::string>
// 		Map_Node(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value,
// 			 map_func_nip_t>::type _func,
// 			 T _name,
// 			 RuntimeContext _context,
// 			 closing_func_t _closing_func):
// 			func_nip(_func),
// 			name(_name),
// 			isIP(false),
// 			isRich(false),
// 			context(_context),
// 			closing_func(_closing_func)
// 		{}

// 		// Constructor IV
// 		template <typename T=std::string>
// 		Map_Node(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value,
// 			 rich_map_func_nip_t>::type _func,
// 			 T _name,
// 			 RuntimeContext _context,
// 			 closing_func_t _closing_func):
// 			rich_func_nip(_func),
// 			name(_name),
// 			isIP(false),
// 			isRich(true),
// 			context(_context),
// 			closing_func(_closing_func)
// 		{}

// 		// svc_init method (utilized by the FastFlow runtime)
// 		int svc_init()
// 		{
// #if defined(LOG_DIR)
// 			logfile = new std::ofstream();
// 			name += "_node_"
// 				+ std::to_string(ff::ff_node_t<tuple_t,result_t>::get_my_id())
// 				+ ".log";
// 			std::string filename = std::string(STRINGIFY(LOG_DIR))
// 				+ "/" + name;
// 			logfile->open(filename);
// #endif
// 			return 0;
// 		}

// 		// svc method (utilized by the FastFlow runtime)
// 		result_t *svc(tuple_t *t)
// 		{
// #if defined (LOG_DIR)
// 			startTS = current_time_nsecs();
// 			if (rcvTuples == 0)
// 				startTD = current_time_nsecs();
// 			rcvTuples++;
// #endif
// 			result_t *r;
// 			// in-place version
// 			if (isIP) {
// 				if (!isRich)
// 					func_ip(*t);
// 				else
// 					rich_func_ip(*t, context);
// 				r = reinterpret_cast<result_t *>(t);
// 			}
// 			else {
// 				r = new result_t();
// 				if (!isRich)
// 					func_nip(*t, *r);
// 				else
// 					rich_func_nip(*t, *r, context);
// 				delete t;
// 			}
// #if defined(LOG_DIR)
// 			endTS = current_time_nsecs();
// 			endTD = current_time_nsecs();
// 			double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
// 			avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
// 			double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
// 			avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
// 			startTD = current_time_nsecs();
// #endif
// 			return r;
// 		}

// 		// svc_end method (utilized by the FastFlow runtime)
// 		void svc_end()
// 		{
// 			// call the closing function
// 			closing_func(context);
// #if defined (LOG_DIR)
// 			std::ostringstream stream;
// 			stream << "************************************LOG************************************\n";
// 			stream << "No. of received tuples: " << rcvTuples << "\n";
// 			stream << "Average service time: " << avg_ts_us << " usec \n";
// 			stream << "Average inter-departure time: " << avg_td_us << " usec \n";
// 			stream << "***************************************************************************\n";
// 			*logfile << stream.str();
// 			logfile->close();
// 			delete logfile;
// #endif
// 		}
// 	};

public:
	/**
	 *  \brief Constructor I
	 *
	 *  \param _func function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func) :
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		// std::vector<ff_node *> w;
		// for (std::size_t i=0; i<_pardegree; i++) {
		// 	auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
		// 	w.push_back(seq);
		// }
		// add emitter
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		// ff::ff_farm::add_workers(w);
		// add default collector
		// ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		// ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor II
	 *
	 *  \param _func function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func) :
	    //routing_func_t _routing_func):
		keyed(true)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		// std::vector<ff_node *> w;
		// for (std::size_t i=0; i<_pardegree; i++) {
		// 	auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
		// 	w.push_back(seq);
		// }
		// add emitter
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
		// add workers
		// ff::ff_farm::add_workers(w);
		// add default collector
		// ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		// ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor III
	 *
	 *  \param _func rich function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value,
	    rich_map_func_ip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func):
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		// std::vector<ff_node *> w;
		// for (std::size_t i=0; i<_pardegree; i++) {
		// 	auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
		// 	w.push_back(seq);
		// }
		// add emitter
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		// ff::ff_farm::add_workers(w);
		// add default collector
		// ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		// ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor IV
	 *
	 *  \param _func rich function to be executed on each input tuple (in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, rich_map_func_ip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func) :
		//routing_func_t _routing_func):
		keyed(true)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		std::vector<ff_node *> w;
		for (std::size_t i=0; i<_pardegree; i++) {
			auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
			w.push_back(seq);
		}
		// add emitter
		ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
		// add workers
		ff::ff_farm::add_workers(w);
		// add default collector
		ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor V
	 *
	 *  \param _func function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func):
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		// std::vector<ff_node *> w;
		// for (std::size_t i=0; i<_pardegree; i++) {
		// 	auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
		// 	w.push_back(seq);
		// }
		// add emitter
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		// ff::ff_farm::add_workers(w);
		// add default collector
		// ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		// ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor VI
	 *
	 *  \param _func function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	// template <typename T=std::size_t>
	// Map(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
	//     T _pardegree,
	//     std::string _name,
	//     closing_func_t _closing_func) :
	//     // routing_func_t _routing_func):
	// 	keyed(true)
	// {
	// 	// check the validity of the parallelism degree
	// 	if (_pardegree == 0) {
	// 		std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
	// 		exit(EXIT_FAILURE);
	// 	}
	// 	// vector of Map_Node
	// 	std::vector<ff_node *> w;
	// 	for (std::size_t i=0; i<_pardegree; i++) {
	// 		auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
	// 		w.push_back(seq);
	// 	}
	// 	// add emitter
	// 	ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
	// 	// add workers
	// 	ff::ff_farm::add_workers(w);
	// 	// add default collector
	// 	ff::ff_farm::add_collector(nullptr);
	// 	// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
	// 	ff::ff_farm::cleanup_all();
	// }

	/**
	 *  \brief Constructor VII
	 *
	 *  \param _func rich function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 */
	template <typename T=std::size_t>
	Map(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, rich_map_func_nip_t>::type _func,
	    T _pardegree,
	    std::string _name,
	    closing_func_t _closing_func):
		keyed(false)
	{
		// check the validity of the parallelism degree
		if (_pardegree == 0) {
			std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
			exit(EXIT_FAILURE);
		}
		// vector of Map_Node
		// std::vector<ff_node *> w;
		// for (std::size_t i=0; i<_pardegree; i++) {
		// 	auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
		// 	w.push_back(seq);
		// }
		// add emitter
		// ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_pardegree));
		// add workers
		// ff::ff_farm::add_workers(w);
		// add default collector
		// ff::ff_farm::add_collector(nullptr);
		// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
		// ff::ff_farm::cleanup_all();
	}

	/**
	 *  \brief Constructor VIII
	 *
	 *  \param _func rich function to be executed on each input tuple (not in-place version)
	 *  \param _pardegree parallelism degree of the Map operator
	 *  \param _name string with the unique name of the Map operator
	 *  \param _closing_func closing function
	 *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
	 */
	template <typename T=std::size_t>
	// Map(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, rich_map_func_nip_t>::type _func,
	//     T _pardegree,
	//     std::string _name,
	//     closing_func_t _closing_func) :
	//     // routing_func_t _routing_func):
	// 	keyed(true)
	// {
	// 	// check the validity of the parallelism degree
	// 	if (_pardegree == 0) {
	// 		std::cerr << RED << "WindFlow Error: Map has parallelism zero" << DEFAULT << std::endl;
	// 		exit(EXIT_FAILURE);
	// 	}
	// 	// vector of Map_Node
	// 	std::vector<ff_node *> w;
	// 	for (std::size_t i=0; i<_pardegree; i++) {
	// 		auto *seq = new Map_Node(_func, _name, RuntimeContext(_pardegree, i), _closing_func);
	// 		w.push_back(seq);
	// 	}
	// 	// add emitter
	// 	ff::ff_farm::add_emitter(new Standard_Emitter<tuple_t>(_routing_func, _pardegree));
	// 	// add workers
	// 	ff::ff_farm::add_workers(w);
	// 	// add default collector
	// 	ff::ff_farm::add_collector(nullptr);
	// 	// when the Map will be destroyed we need aslo to destroy the emitter, workers and collector
	// 	ff::ff_farm::cleanup_all();
	// }

	/**
	 *  \brief Check whether the Map has been instantiated with a key-based distribution or not
	 *  \return true if the Map is configured with keyBy
	 */
	bool isKeyed() const
	{
		return keyed;
	}

	// svc_init method (utilized by the FastFlow runtime)
	int svc_init()
	{
#if defined(LOG_DIR)
		logfile = new std::ofstream();
		name += "_node_"
			+ std::to_string(ff::ff_node_t<tuple_t,result_t>::get_my_id())
			+ ".log";
		std::string filename = std::string(STRINGIFY(LOG_DIR))
			+ "/" + name;
		logfile->open(filename);
#endif
		return 0;
	}
};

} // namespace wf

#endif