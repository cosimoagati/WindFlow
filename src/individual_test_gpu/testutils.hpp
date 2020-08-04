#ifndef TESTUTIlS_HPP
#define TESTUTIlS_HPP

#include <iostream>
#include "../../wf/windflow_gpu.hpp"
#include "../../wf/windflow.hpp"

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
	std::chrono::time_point<std::chrono::steady_clock> start_time {std::chrono::steady_clock::now()};
	std::chrono::time_point<std::chrono::steady_clock> end_time {start_time + std::chrono::seconds {10}};
#endif
public:
	int svc_init() {
		std::cout << "Initializing source..." << std::endl;
		return 0;
	}
	tuple_t *svc(tuple_t *) {
#ifdef PERFORMANCE_TEST
		const auto current_time = std::chrono::steady_clock::now();
		if (current_time >= end_time)
			return this->EOS;
#else
		if (counter > LIMIT)
			return this->EOS;
#endif
		// Generate them all with key 0 for simplicity.
		const auto t = new tuple_t {0, counter, counter, counter};
		++counter;
		return t;
	}
	void svc_end() { std::cout << "Source closing..." << std::endl; }
};

template<typename tuple_t>
class Sink : public ff::ff_node_t<tuple_t, tuple_t> {
public:
	int svc_init() {
		std::cout << "Initializing sink..." << std::endl;
		return 0;
	}
	tuple_t *svc(tuple_t *const t) {
		output_stream << t->value << '\n';
		delete t;
		return this->GO_ON;
	}
	void svc_end() { std::cout << "Sink closing..." << std::endl; }
};

void closing_func(wf::RuntimeContext &) {}

int routing_func(const size_t k, const size_t n) { return k % n; }

#endif
