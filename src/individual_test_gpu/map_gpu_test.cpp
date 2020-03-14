#include "../../wf/windflow_gpu.hpp"

__device__ void square(int x) { return x * x; }

void closing_func(RuntimeContext &r) { (void) r; }

int main(int argc, char *argv[])
{
	auto map = MapGPU<int, int, decltype(square)> {square, 4, "gino",
						       closing_func};
	return 0;
}
