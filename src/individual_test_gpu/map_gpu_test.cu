#include <cstdint>
#include <iostream>
#include <functional>
#include "../../wf/windflow_gpu.hpp"

using namespace std;
using namespace wf;

struct tuple_t
{
	size_t key;
	uint64_t id;
	uint64_t ts;
	int64_t value;

	tuple_t(size_t _key,
		uint64_t _id,
		uint64_t _ts,
		int64_t _value):
		key(_key),
		id(_id),
		ts(_ts),
		value(_value)
	{}

	tuple_t():
		key(0),
		id(0),
		ts(0),
		value(0)
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

void
closing_func(RuntimeContext &r) { (void) r; }

int main(int argc, char *argv[])
{
	auto square = [] __host__ __device__ (tuple_t &x)
		{
		 x.value = x.value * x.value + 4;
		};
	auto another_square = [] __host__ __device__ (const tuple_t &x, tuple_t &y)
		{
		 y.value = x.value * x.value;
		};
	auto map = MapGPU<tuple_t, tuple_t, decltype(square)> {square, 4, "gino",
							       closing_func};
	cout << "Now outside of constructor" << endl;
	cout << wf::is_invocable<decltype(square), tuple_t &>::value << endl;
	cout << wf::is_invocable<decltype(square), const tuple_t &, tuple_t &>::value << endl;
	return 0;
}
