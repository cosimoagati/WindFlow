#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <tuple>

#define TRACE_WINDFLOW log
// #include "../../wf/builders.hpp"
#include "../../wf/windflow.hpp"
#include "../../wf/windflow_gpu.hpp"

#define TRIVIAL_TEST

using namespace std;
using namespace std::chrono;
using namespace ff;
using namespace wf;

// Global is ugly, but simple
ofstream output_stream {"output.txt"};

struct tuple_t {
	size_t key;
	uint64_t id;
	uint64_t ts;
	int64_t value;

	tuple_t(size_t key, uint64_t id, uint64_t ts, int64_t value)
		: key {key}, id {id}, ts {ts}, value{value}
	{}
	tuple_t() : key {0}, id {0}, ts {0}, value {0} {}

	tuple<size_t, uint64_t, uint64_t> getControlFields() const {
		return tuple<size_t, uint64_t, uint64_t>(key, id, ts);
	}
	void setControlFields(size_t key, uint64_t id, uint64_t ts) {
		this->key = key;
		this->id = id;
		this->ts = ts;
	}
};

template<typename tuple_t>
class Source : public ff_node_t<tuple_t, tuple_t> {
	static constexpr auto LIMIT = 1000;
	long counter {0};
#ifndef TRIVIAL_TEST
	time_point<steady_clock> start_time {steady_clock::now()};
	time_point<steady_clock> end_time = start_time + seconds {10};
#endif
public:
	int svc_init() {
		cout << "Initializing source..." << endl;
		return 0;
	}
	tuple_t *svc(tuple_t *) {
#ifdef TRIVIAL_TEST
		if (counter > LIMIT) {
			return this->EOS;
		}
#else
		const auto current_time = steady_clock::now();
		if (current_time >= end_time) {
			return this->EOS;
		}
#endif
		// Generate them all with key 0 for simplicity.
		const auto t = new tuple_t {0, counter, counter, counter};
		++counter;
		return t;
	}
	void svc_end() { cout << "Source closing..." << endl; }
};

template<typename tuple_t>
class Sink : public ff_node_t<tuple_t, tuple_t> {
public:
	int svc_init() {
		cout << "Initializing sink..." << endl;
		return 0;
	}
	tuple_t *svc(tuple_t *const t) {
		output_stream << t->value << '\n';
		delete t;
		return this->GO_ON;
	}
	void svc_end() { cout << "Sink closing..." << endl; }
};

void closing_func(RuntimeContext &) {}

int routing_func(const size_t k, const size_t n) { return k % n; }

int test_gpu() {
	const auto drop_if_odd = [] __host__ __device__ (const tuple_t &x,
							 bool &mask) {
		mask = x.value % 2 == 0;
	};
	ff_pipeline ip_pipe;
	ip_pipe.add_stage(::Source<tuple_t> {});
	FilterGPU<tuple_t, decltype(drop_if_odd)> filter {drop_if_odd,
							  1, "filter"};
	ip_pipe.add_stage(&filter);
	ip_pipe.add_stage(::Sink<tuple_t> {});

	if (ip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		std::exit(EXIT_FAILURE);
	}
}

int test_cpu() {
	// TODO
	//output_stream << "Testing \"default\", CPU Filter..." << endl;
}

int main() {
	test_gpu();
	test_cpu();
	return 0;
}
