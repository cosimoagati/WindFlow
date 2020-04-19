#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <functional>
#include "../../wf/builders.hpp"
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

	void setControlFields(size_t _key, uint64_t _id, uint64_t _ts) {
		key = _key;
		id = _id;
		ts = _ts;
	}
};

template<typename tuple_t>
class Source : public ff_node_t<tuple_t, tuple_t> {
	static constexpr auto LIMIT = 1000;
	long counter {0};
#ifndef TRIVIAL_TEST
	time_point<steady_clock> start_time {steady_clock::now()};
	time_point<steady_clock> end_time = start_time + seconds {60}
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

	tuple_t *svc(tuple_t *t) {
		output_stream << t->value << '\n';
		delete t;
		return this->GO_ON;
	}

	void svc_end() { cout << "Sink closing..." << endl; }
};

void closing_func(RuntimeContext &) {}

int routing_func(size_t k, size_t n) { return k % n; }

int main(int argc, char *argv[]) {
	auto square = [] __host__ __device__ (tuple_t &x)
		{
		 x.value = x.value * x.value;
		};
	auto another_square = [] __host__ __device__ (const tuple_t &x, tuple_t &y)
		{
		 y.value = x.value * x.value;
		};
	auto verify_order = [] __host__ __device__ (tuple_t &t, char *scratchpad,
						    std::size_t size)
		{
		 // NB: the first tuple must have value 0!
		 assert(size >= sizeof(int));
		 assert(scratchpad != nullptr);
		 char x = scratchpad[0];
		 const auto prev_val = static_cast<int>(*scratchpad);
		 if (t.value == -1 || (t.value != 0 && t.value <= prev_val)) {
			 t.value = -1;
		 }
		 *reinterpret_cast<int *>(scratchpad) = t.value;
		};
	auto verify_order_nip = [] __host__ __device__ (const tuple_t &t,
							tuple_t &r,
							char *scratchpad,
							std::size_t size)
		{
		 // NB: the first tuple must have value 0!
		 assert(size >= sizeof(int));
		 assert(scratchpad != nullptr);
		 const auto prev_val = static_cast<int>(*scratchpad);
		 if (t.value == -1 || (t.value != 0 && t.value <= prev_val)) {
			 r.value = -1;
		 }
		 *reinterpret_cast<int *>(scratchpad) = r.value;
		};

	ff_pipeline ip_pipe;
	ip_pipe.add_stage(::Source<tuple_t> {});
	auto ip_map = MapGPU_Builder<decltype(square)> {square}.withParallelism(1).build_ptr();
	ip_pipe.add_stage(ip_map);
	ip_pipe.add_stage(::Sink<tuple_t> {});

	if (ip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		return -1;
	}
	delete ip_map; // Sadly needed since copy elision for builders only
		       // works from C++17 onwards.

	output_stream << "In-pace pipeline finished, now testing non in-place version..."
		      << endl;
	ff_pipeline nip_pipe;
	nip_pipe.add_stage(::Source<tuple_t> {});
	auto nip_map = MapGPU_Builder<decltype(another_square)> {another_square}.withParallelism(1)
											 .build_ptr();
	nip_pipe.add_stage(nip_map);
	nip_pipe.add_stage(::Sink<tuple_t> {});

	if (nip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		return -1;
	}
	delete nip_map; // Same notce above.

	output_stream << "Non in-pace pipeline finished, now testing in-place keyed version..."
		      << endl;
	ff_pipeline ip_keyed_pipe;
	ip_keyed_pipe.add_stage(::Source<tuple_t> {});

	// TODO: Builder still needs a correct get_tuple_t meta_utils function!
	// auto ip_keyed_map = MapGPU_Builder<decltype(verify_order)> {verify_order}.withParallelism(1)
	// 										  .enable_KeyBy()
	// 										  .build_ptr();
	MapGPU<tuple_t, tuple_t, decltype(verify_order)> ip_keyed_map {verify_order,
			1, "gino", routing_func, 256, 256, sizeof(int)};
	ip_keyed_pipe.add_stage(&ip_keyed_map);
	ip_keyed_pipe.add_stage(::Sink<tuple_t> {});
	if (ip_keyed_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline");
		return -1;
	}
	return 0;
}
