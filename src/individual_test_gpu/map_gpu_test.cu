#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <tuple>

#define TRACE_WINDFLOW log
#include "../../wf/builders.hpp"
#include "../../wf/windflow_gpu.hpp"
#include "../../wf/windflow.hpp"

// #define PERFORMANCE_TEST

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
#ifdef PERFORMANCE_TEST
	time_point<steady_clock> start_time {steady_clock::now()};
	time_point<steady_clock> end_time = start_time + seconds {10};
#endif
public:
	int svc_init() {
		cout << "Initializing source..." << endl;
		return 0;
	}
	tuple_t *svc(tuple_t *) {
#ifdef PERFORMANCE_TEST
		const auto current_time = steady_clock::now();
		if (current_time >= end_time) {
			return this->EOS;
		}
#else
		if (counter > LIMIT) {
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

void test_gpu() {
#ifdef PERFORMANCE_TEST
	const auto ip_keyless_func = [] __host__ __device__ (tuple_t &x) {
		// Deliberately long-to-compute function.  Cheap functions that
		// cost less than buffering are not representative.
		auto divide = false;
		for (auto i = 0; i < 1000; ++i) {
			if (divide && x.value != 0) {
				x.value /= x.value;
			} else {
				x.value *= x.value;
			}
			divide = !divide;
		}
	};
#else
	const auto ip_keyless_func = [] __host__ __device__ (tuple_t &x) {
		x.value *= x.value;
	};
#endif
#ifndef PERFORMANCE_TEST
	const auto nip_keyless_func = [] __host__ __device__ (const tuple_t &x,
							      tuple_t &y) {
		y.value = x.value * x.value;
	};
	const auto ip_keyed_func = [] __host__ __device__ (tuple_t &t,
							   char *const scratchpad,
							   const std::size_t size) {
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
	const auto nip_keyed_func = [] __host__ __device__ (const tuple_t &t,
							    tuple_t &r,
							    char *const scratchpad,
							    const std::size_t size) {
		// NB: the first tuple must have value 0!
		assert(size >= sizeof(int));
		assert(scratchpad != nullptr);
		const auto prev_val = static_cast<int>(*scratchpad);
		if (t.value == -1 || (t.value != 0 && t.value <= prev_val)) {
			r.value = -1;
		}
		*reinterpret_cast<int *>(scratchpad) = r.value;
	};
#endif
	ff_pipeline ip_pipe;
	ip_pipe.add_stage(::Source<tuple_t> {});
	auto ip_map = MapGPU_Builder<decltype(ip_keyless_func)> {ip_keyless_func}
		.withName("gpu_ip_map")
		.withParallelism(1)
		.build_ptr();
	ip_pipe.add_stage(ip_map);
	ip_pipe.add_stage(::Sink<tuple_t> {});

	if (ip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		std::exit(EXIT_FAILURE);
	}
	delete ip_map; // Sadly needed since copy elision for builders only
		       // works from C++17 onwards.

#ifndef PERFORMANCE_TEST
	output_stream << "In-pace pipeline finished, now testing non in-place version..."
		      << endl;
	ff_pipeline nip_pipe;
	nip_pipe.add_stage(::Source<tuple_t> {});
	auto nip_map = MapGPU_Builder<decltype(nip_keyless_func)> {nip_keyless_func}
		.withName("gpu_nip_map")
		.withParallelism(1)
		.build_ptr();
	nip_pipe.add_stage(nip_map);
	nip_pipe.add_stage(::Sink<tuple_t> {});

	if (nip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		std::exit(EXIT_FAILURE);
	}
	delete nip_map; // Same notce above.

	output_stream << "Non in-pace pipeline finished, now testing in-place keyed version..."
		      << endl;
	ff_pipeline ip_keyed_pipe;
	ip_keyed_pipe.add_stage(::Source<tuple_t> {});

	// TODO: Builder still needs a correct get_tuple_t meta_utils function!
	// auto ip_keyed_map = MapGPU_Builder<decltype(ip_keyed_func)> {ip_keyed_func}.withParallelism(1)
	// 										  .enable_KeyBy()
	// 										  .build_ptr();
	MapGPU<tuple_t, tuple_t, decltype(ip_keyed_func)> ip_keyed_map {ip_keyed_func,
			1, "gpu_ip_keyed_map", routing_func, 256, 256, sizeof(int)};
	ip_keyed_pipe.add_stage(&ip_keyed_map);
	ip_keyed_pipe.add_stage(::Sink<tuple_t> {});
	if (ip_keyed_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline");
		std::exit(EXIT_FAILURE);
	}
#endif
}

#ifdef PERFORMANCE_TEST
void test_cpu() {
	output_stream << "Testing \"default\", CPU Map..." << endl;

	const auto ip_keyless_func = [] (tuple_t &x) {
		// Deliberately long-to-compute function.  Cheap functions that
		// cost less than buffering are not representative.
		auto divide = false;
		for (auto i = 0; i < 1000; ++i) {
			if (divide && x.value != 0) {
				x.value /= x.value;
			} else {
				x.value *= x.value;
			}
			divide = !divide;
		}
	};
	const auto nip_keyless_func = [] (const tuple_t &x, tuple_t &y) {
		y.value = x.value * x.value;
	};
	const auto ip_keyed_func = [] (tuple_t &t, char *const scratchpad,
				      const std::size_t size) {
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
	const auto ip_keyed_func_nip = [] (const tuple_t &t, tuple_t &r, char *scratchpad,
					  std::size_t size) {
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
	auto ip_map = Map_Builder<decltype(ip_keyless_func)> {ip_keyless_func}
		.withName("cpu_ip_map")
		.withParallelism(1)
		.build_ptr();
	ip_pipe.add_stage(ip_map);
	ip_pipe.add_stage(::Sink<tuple_t> {});

	if (ip_pipe.run_and_wait_end() < 0) {
		error("Error while running pipeline.");
		std::exit(EXIT_FAILURE);
	}
	delete ip_map; // Sadly needed since copy elision for builders only
		       // works from C++17 onwards.
}
#endif

int main(int argc, char *argv[]) {
	test_gpu();
#ifdef PERFORMANCE_TEST
	test_cpu();
#endif
	return 0;
}
