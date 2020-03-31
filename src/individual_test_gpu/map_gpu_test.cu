#include <cstdint>
#include <iostream>
#include <fstream>
#include <functional>
#include "../../wf/windflow_gpu.hpp"

using namespace std;
using namespace ff;
using namespace wf;

// Global is ugly, but simple
ofstream output_stream {"output.txt"};

struct tuple_t
{
	size_t key;
	uint64_t id;
	uint64_t ts;
	int64_t value;

	tuple_t(size_t key, uint64_t id, uint64_t ts, int64_t value)
		: key {key}, id {id}, ts {ts}, value{value}
	{}

	tuple_t() : key {0}, id {0}, ts {0}, value {0}
	{}

	tuple<size_t, uint64_t, uint64_t> getControlFields() const
	{
		return tuple<size_t, uint64_t, uint64_t>(key, id, ts);
	}

	void setControlFields(size_t _key, uint64_t _id, uint64_t _ts)
	{
		key = _key;
		id = _id;
		ts = _ts;
	}
};

template<typename tuple_t>
class Source : public ff_node_t<tuple_t, tuple_t>
{
	static constexpr auto LIMIT = 1000;
	long counter {0};
public:
	int
	svc_init()
	{
		cout << "Initializing source..." << endl;
		return 0;
	}

	tuple_t *
	svc(tuple_t *)
	{
		if (counter > LIMIT)
			return this->EOS;
		const auto t = new tuple_t {counter, counter, counter, counter};
		++counter;
		return t;
	}

	void
	svc_end()
	{
		cout << "Source closing..." << endl;
	}
};

template<typename tuple_t>
class Sink : public ff_node_t<tuple_t, tuple_t>
{
public:
	int
	svc_init()
	{
		cout << "Initializing sink..." << endl;
		return 0;
	}

	tuple_t *
	svc(tuple_t *t)
	{
		output_stream << t->value << '\n';
		delete t;
		return this->GO_ON;
	}

	void svc_end()
	{
		cout << "Sink closing..." << endl;
	}
};

void closing_func(RuntimeContext &) {}

int main(int argc, char *argv[])
{
	auto square = [] __host__ __device__ (tuple_t &x)
		{
		 x.value = x.value * x.value;
		};
	auto another_square = [] __host__ __device__ (const tuple_t &x, tuple_t &y)
		{
		 y.value = x.value * x.value;
		};


	ff_pipeline ip_pipe;
	ip_pipe.add_stage(::Source<tuple_t> {});
	MapGPU<tuple_t, tuple_t, decltype(square)> ip_map {square, 4, "gino",
			                                   closing_func};
	ip_pipe.add_stage(&ip_map);
	ip_pipe.add_stage(::Sink<tuple_t> {});

	if (ip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		return -1;
	}

	output_stream << "In-pace pipeline finished, now testing non in-place version..."
		      << endl;

	ff_pipeline nip_pipe;
	nip_pipe.add_stage(::Source<tuple_t> {});
	MapGPU<tuple_t, tuple_t, decltype(another_square)> nip_map{another_square,
								   1, "gino", closing_func};
	nip_pipe.add_stage(&nip_map);
	nip_pipe.add_stage(::Sink<tuple_t> {});

	if (nip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		return -1;
	}

	return 0;
}
