#include <functional>
#include <iostream>
#include <cmath>

// This is an example to show how you can pass an arbitrary
// __device__ function to a CUDA kernel using multiple types, such
// as a function pointer or a lambda expression.


using namespace std;

// Doesn't work if host_kernel calls THIS function!
__host__ __device__ float
square(float x) { return x * x; }

// template<typename F_t>
// __device__ void
// perform_op(F_t f, float *x, float *y, const int n)
// {
// 	for (int i = 0; i < n; ++i)
// 		y[i] = f(x[i]);
// }

template<typename F_t>
__global__ void
host_kernel(float *x, float *y, const int n, F_t f)
{
	// Double numbers and then square them!
	// const auto double_num = [] __device__ (float x) { return 2 * x; };
	// perform_op<decltype(double_num)>(double_num, x, y, n);
	// perform_op<decltype(square)>(square, y, y, n);

	for (int i = 0; i < n; ++i)
		y[i] = f(x[i]);
}

// template<typename F_t>
// void
// call_kernel(F_t kernel, float *x, float *y, const int n)
// {
// 	kernel<<<1, 1>>>(x, y, n);
// }

int
main(void)
{
	int N = 1<<20;
	float *x, *y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	for (int i = 0; i < N; i++)
		x[i] = static_cast<float>(i);
	// This lambda function, however, works!
	auto square = [] __device__ (float x) {return x * x; };
	host_kernel<decltype(square)><<<1, 1>>>(x, y, N, square);
	cudaDeviceSynchronize();
	// call_kernel(host_kernel, x, y, N);
	// cudaDeviceSynchronize(); // Wait for GPU before accessing on host.

	for (int i = 0; i < 100; i++) // Only verify up to 99 for simplicity.
		cout << y[i] << "\n";
	cudaFree(x); // Free memory
	cudaFree(y);

	return 0;
}