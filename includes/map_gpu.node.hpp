#ifndef MAP_GPU_NODE_H
#define MAP_GPU_NODE_H

#include <vector>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <basic.hpp>
#include <context.hpp>
#include <standard_nodes.hpp>

namespace wf
{
template<typename tuple_t, typename result_t>
class Map_Node: public ff::ff_node_t<tuple_t, result_t>
{
	// Shamelessly stolen from map.hpp
	/// type of the map function (in-place version)
	using map_func_ip_t = std::function<void(tuple_t &)>;
	/// type of the rich map function (in-place version)
	using rich_map_func_ip_t = std::function<void(tuple_t &, RuntimeContext &)>;
	/// type of the map function (not in-place version)
	using map_func_nip_t = std::function<void(const tuple_t &, result_t &)>;
	/// type of the rich map function (not in-place version)
	using rich_map_func_nip_t = std::function<void(const tuple_t &, result_t &, RuntimeContext &)>;
	/// type of the closing function
	using closing_func_t = std::function<void(RuntimeContext &)>;
	/// type of the function to map the key hashcode onto an identifier starting from zero to pardegree-1
	using routing_func_t = std::function<std::size_t(std::size_t, std::size_t)>;

	static constexpr auto max_buffered_tuples = 256;
	std::vector<tuple_t *> tuple_buffer;
	std::vector<result_t *> result_buffer;

	map_func_ip_t func_ip; // in-place map function
	rich_map_func_ip_t rich_func_ip; // in-place rich map function
	map_func_nip_t func_nip; // not in-place map function
	rich_map_func_nip_t rich_func_nip; // not in-place rich map function
	closing_func_t closing_func; // closing function
	std::string name; // string of the unique name of the operator
	bool isIP; // flag stating if the in-place map function should be used (otherwise the not in-place version)
	bool isRich; // flag stating whether the function to be used is rich (i.e. it receives the RuntimeContext object)
	RuntimeContext context; // RuntimeContext
#if defined(LOG_DIR)
	unsigned long rcvTuples = 0;
	double avg_td_us = 0;
	double avg_ts_us = 0;
	volatile unsigned long startTD, startTS, endTD, endTS;
	std::ofstream *logfile = nullptr;
#endif

public:
	// Constructor I
	template <typename T=std::string>
	Map_Node(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, map_func_ip_t>::type _func,
		 T _name,
		 RuntimeContext _context,
		 closing_func_t _closing_func):
		func_ip(_func),
		name(_name),
		isIP(true),
		isRich(false),
		context(_context),
		closing_func(_closing_func)
	{}

	// Constructor II
	template <typename T=std::string>
	Map_Node(typename std::enable_if<std::is_same<T,T>::value && std::is_same<tuple_t,result_t>::value, rich_map_func_ip_t>::type _func,
		 T _name,
		 RuntimeContext _context,
		 closing_func_t _closing_func):
		rich_func_ip(_func),
		name(_name),
		isIP(true),
		isRich(true),
		context(_context),
		closing_func(_closing_func)
	{}

	// Constructor III
	template <typename T=std::string>
	Map_Node(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value, map_func_nip_t>::type _func,
		 T _name,
		 RuntimeContext _context,
		 closing_func_t _closing_func):
		func_nip(_func),
		name(_name),
		isIP(false),
		isRich(false),
		context(_context),
		closing_func(_closing_func)
	{}

	// Constructor IV
	template <typename T=std::string>
	Map_Node(typename std::enable_if<std::is_same<T,T>::value && !std::is_same<tuple_t,result_t>::value,
		 rich_map_func_nip_t>::type _func,
		 T _name,
		 RuntimeContext _context,
		 closing_func_t _closing_func):
		rich_func_nip(_func),
		name(_name),
		isIP(false),
		isRich(true),
		context(_context),
		closing_func(_closing_func)
	{}

	// svc_init method (utilized by the FastFlow runtime)
	int svc_init()
	{
#if defined(LOG_DIR)
		logfile = new std::ofstream();
		name += "_node_" + std::to_string(ff::ff_node_t<tuple_t, result_t>::get_my_id()) + ".log";
		std::string filename = std::string(STRINGIFY(LOG_DIR)) + "/" + name;
		logfile->open(filename);
#endif
		return 0;
	}

	// svc method (utilized by the FastFlow runtime)
	result_t *svc(tuple_t *t)
	{
#if defined (LOG_DIR)
		startTS = current_time_nsecs();
		if (rcvTuples == 0)
			startTD = current_time_nsecs();
		rcvTuples++;
#endif
		result_t *r;
		const auto &output_buffer = isIP
			? tuple_buffer
			: result_buffer;
		// in-place version
		if (tuple_buffer.size() < max_buffered_tuples - 1) {
			tuple_buffer.push_back(t);
			return GO_ON;
		} else {
			if (isIP) {
				if (!isRich)
					func_ip(*t);
				else
					rich_func_ip(*t, context);
				r = reinterpret_cast<result_t *>(t);
				for (const auto &t : tuple_buffer)
					ff_send_out(t);
			} else {
				r = new result_t();
				if (!isRich)
					func_nip(*t, *r);
				else
					rich_func_nip(*t, *r, context);
				for (const auto &t : tuple_buffer)
					delete t;
				for (const auto &r : result_buffer)
					ff_send_out(r);
			}
			for (const auto &r : result_buffer)
				ff_send_out(r);
		}
#if defined(LOG_DIR)
		endTS = current_time_nsecs();
		endTD = current_time_nsecs();
		double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
		avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
		double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
		avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
		startTD = current_time_nsecs();
#endif
	}

	// svc_end method (utilized by the FastFlow runtime)
	void svc_end()
	{
		// call the closing function
		closing_func(context);
#if defined (LOG_DIR)
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
} // namespace wf

#endif
