/******************************************************************************
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 ******************************************************************************
 */

/**
 *  @file    map_gpu_utils.hpp
 *  @author  Cosimo Agati
 *  @date    08/01/2019
 *
 *  @brief Various utilities used by GPU operators, including RAII types to
 *  facilitate resource management.
 *
 *  @section MapGPU_Node (Description)
 */

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

#include "basic.hpp"

namespace wf {

template<typename F, typename... Args>
struct is_invocable :
		std::is_constructible<std::function<void(Args ...)>,
				      std::reference_wrapper<typename std::remove_reference<F>::type>>
{};

/*
 * This struct contains information required to compute a tuple in the
 * keyed/stateful case: the hash is used to ensure tuples with the same key are
 * always computed by the same thread, and a pointer to the respective
 * scratchpad is stored.
 */
struct TupleState {
	std::size_t hash;
	char *scratchpad;
};

/*
 * This struct is useful to pass a tuple buffer allocated on GPU memory directly
 * to another GPU operator, in order to avoid re-buffering.
 */
template<typename tuple_t>
struct GPUBufferHandle {
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;

	tuple_t *buffer;
	std::size_t size;
	key_t *key_buffer;
	std::size_t keys_size;
};

/*
 * GPU buffer abstraction, managing allocation, resize and deletion.
 */
template<typename T>
class GPUBuffer {
	T *buffer_ptr;
	std::size_t buffer_size;
	std::size_t allocated_size;
public:
	GPUBuffer(const std::size_t size)
		: buffer_size {size}, allocated_size {size}
	{
		if (size > 0) {
			const auto status = cudaMalloc(&buffer_ptr, size * sizeof *buffer_ptr);
			assert(status == cudaSuccess);
		} else {
			buffer_ptr = nullptr;
		}
	}

	GPUBuffer() : buffer_ptr {nullptr}, buffer_size {0}, allocated_size {0} {}

	~GPUBuffer() {
		if (buffer_ptr != nullptr) {
			const auto status = cudaFree(buffer_ptr);
			assert(status == cudaSuccess);
		}
	}

	GPUBuffer(const GPUBuffer &other)
		: buffer_size {other.buffer_size}, allocated_size {other.allocated_size}
	{
		if (other.buffer_ptr != nullptr && other.buffer_size > 0) {
			const auto status = cudaMalloc(&buffer_ptr, buffer_size * sizeof *buffer_ptr);
			assert(status == cudaSuccess);
			std::copy(other.buffer_ptr, other.buffer_ptr + buffer_size, buffer_ptr);
 		} else {
			buffer_ptr = nullptr;
		}
	}

	GPUBuffer(GPUBuffer &&other)
		: buffer_ptr {other.buffer_ptr}, buffer_size {other.buffer_size},
		  allocated_size {other.allocated_size}
	{
		other.buffer_ptr = nullptr;
		other.allocated_size = other.buffer_size = 0;
	}

	GPUBuffer &operator=(const GPUBuffer &other) {
		if (buffer_ptr != other.buffer_ptr) {
			// TODO: should resize instead of reallocating...
			if (buffer_ptr != nullptr) {
				const auto status = cudaFree(buffer_ptr);
				assert(status == cudaSuccess);
			}
			// Only copies the relevant elements, even if allocated
			// size of other is greater.
			allocated_size = buffer_size = other.buffer_size;
			const auto status = cudaMalloc(&buffer_ptr, buffer_size * sizeof *buffer_ptr) ;
			assert(status == cudaSuccess);
			std::copy(other.buffer_ptr, other.buffer_ptr + buffer_size, buffer_ptr);
		}
		return *this;
	}

	GPUBuffer &operator=(GPUBuffer &&other) {
		if (buffer_ptr != other.buffer_ptr) {
			if (buffer_ptr != nullptr) {
				const auto status = cudaFree(buffer_ptr);
				assert(status == cudaSuccess);
			}
			buffer_ptr = other.buffer_ptr;
			buffer_size = other.buffer_size;
			allocated_size = other.allocated_size;
			other.buffer_ptr = nullptr;
			other.allocated_size = other.buffer_size = 0;
		}
		return *this;
	}

	T *data() const { return buffer_ptr; }
	std::size_t size() const { return buffer_size; }

	void resize(const std::size_t new_size) {
		if (new_size <= allocated_size) {
			buffer_size = new_size;
			return;
		}
		T *tmp;
		auto status = cudaMalloc(&tmp, new_size * sizeof *tmp);
		assert(status == cudaSuccess);
		if (buffer_ptr != nullptr) {
			status = cudaFree(buffer_ptr);
			assert(status == cudaSuccess);
		}
		buffer_ptr = tmp;
		allocated_size = buffer_size = new_size;
	}
};

template<typename T>
class PinnedCPUBuffer {
	T *buffer_ptr;
	std::size_t buffer_size;
	std::size_t allocated_size;
public:
	PinnedCPUBuffer(const std::size_t size) : buffer_size {size}, allocated_size {size} {
		if (size > 0) {
			const auto status = cudaMallocHost(&buffer_ptr, size * sizeof *buffer_ptr);
			assert(status == cudaSuccess);
		} else {
			buffer_ptr = nullptr;
		}
	}

	PinnedCPUBuffer() : buffer_ptr {nullptr}, buffer_size {0}, allocated_size {0} {}

	~PinnedCPUBuffer() {
		if (buffer_ptr != nullptr) {
			const auto status = cudaFreeHost(buffer_ptr);
			assert(status == cudaSuccess);
		}
	}

	PinnedCPUBuffer(const PinnedCPUBuffer &other)
		: buffer_size {other.buffer_size}, allocated_size {other.allocated_size}
	{
		if (other.buffer_ptr != nullptr && other.buffer_size > 0) {
			const auto status = cudaMallocHost(&buffer_ptr, buffer_size * sizeof *buffer_ptr);
			assert(status == cudaSuccess);
			std::copy(other.buffer_ptr, other.buffer_ptr + buffer_size, buffer_ptr);
		} else {
			buffer_ptr = nullptr;
		}
	}

	PinnedCPUBuffer(PinnedCPUBuffer &&other)
		: buffer_ptr {other.buffer_ptr}, buffer_size {other.buffer_size},
		  allocated_size {other.allocated_size}
	{
		other.buffer_ptr = nullptr;
		other.allocated_size = other.buffer_size = 0;
	}

	PinnedCPUBuffer &operator=(const PinnedCPUBuffer &other) {
		if (buffer_ptr != other.buffer_ptr) {
			if (buffer_ptr != nullptr) {
				const auto status = cudaFreeHost(buffer_ptr);
				assert(status == cudaSuccess);
			}
			// Only copies the relevant elements, even if allocated
			// size of other is greater.
			allocated_size = buffer_size = other.buffer_size;
			const auto status = cudaMallocHost(&buffer_ptr, buffer_size * sizeof *buffer_ptr) ;
			assert(status == cudaSuccess);
			std::copy(other.buffer_ptr, other.buffer_ptr + buffer_size, buffer_ptr);
		}
		return *this;
	}

	PinnedCPUBuffer &operator=(PinnedCPUBuffer &&other) {
		if (buffer_ptr != other.buffer_ptr) {
			if (buffer_ptr != nullptr) {
				const auto status = cudaFreeHost(buffer_ptr);
				assert(status == cudaSuccess);
			}
			buffer_ptr = other.buffer_ptr;
			buffer_size = other.buffer_size;
			allocated_size = other.allocated_size;
			other.allocated_size = other.buffer_size = 0;
			other.buffer_ptr = nullptr;
		}
		return *this;
	}

	T *data() const { return buffer_ptr; }
	std::size_t size() const { return buffer_size; }
	T &operator[](std::size_t i) { return buffer_ptr[i]; }
	const T &operator[](std::size_t i) const { return buffer_ptr[i]; }

	void resize(const std::size_t new_size) {
		if (new_size <= allocated_size) {
			buffer_size = new_size;
			return;
		}
		T *tmp;
		auto status = cudaMallocHost(&tmp, new_size * sizeof *tmp);
		assert(status == cudaSuccess);
		if (buffer_ptr != nullptr) {
			status = cudaFreeHost(buffer_ptr);
			assert(status == cudaSuccess);
		}
		buffer_ptr = tmp;
		allocated_size = buffer_size = new_size;
	}
};

class GPUStream {
	cudaStream_t stream;
public:
	GPUStream() {
		const auto status = cudaStreamCreate(&stream);
		assert(status == cudaSuccess);
	}
	~GPUStream() {
		if (stream != 0) {
			const auto status = cudaStreamDestroy(stream);
			assert(status == cudaSuccess);
		}
	}
	GPUStream(const GPUStream &other) : GPUStream {} {}

	GPUStream(GPUStream &&other) {
		stream = other.stream;
		other.stream = 0;
	}

	GPUStream &operator=(const GPUStream &other) {
		// Do nothing: already have our own independent stream.
		return *this;
	}

	GPUStream &operator=(GPUStream &&other) {
		if (stream != other.stream) {
			stream = other.stream;
			other.stream = 0;
		}
		return *this;
	}

	const cudaStream_t &raw() const { return stream; }
	cudaStream_t &raw() { return stream; }

	void sync() {
		const auto status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
	}
};

inline void failwith(const std::string &err) {
	std::cerr << RED << "WindFlow Error: " << err << DEFAULT_COLOR
		  << std::endl;
	std::exit(EXIT_FAILURE);
}

constexpr auto threads_per_block = 512;

template<typename T>
__global__ void map_to_target(T *const output, T *const input, const std::size_t n, const T target_value) {
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	if (absolute_id < n)
		output[absolute_id] = input[absolute_id] == target_value;
}

template<typename T>
__global__ void prescan(T *const output, T *const input, T *const partial_sums, const std::size_t n,
			const std::size_t pow) {
	extern __shared__ int temp[];
	const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
	const auto block_thread_id = threadIdx.x;

	if (absolute_id < n)
		temp[block_thread_id] = input[absolute_id];
	// TODO: bit shifts should be more efficient...
	for (auto stride = 1; stride <= pow; stride *= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index < 2 * n)
			temp[index] += temp[index - stride];
	}
	for (auto stride = pow / 2; stride > 0; stride /= 2) {
		__syncthreads();
		const auto index = (block_thread_id + 1) * stride * 2 - 1;
		if (index + stride < 2 * n)
			temp[index + stride] += temp[index];
	}
	__syncthreads();
	if (block_thread_id == 0 && blockIdx.x > 0)
		partial_sums[blockIdx.x - 1] = temp[n - 1];
	if (absolute_id < n)
		output[absolute_id] = temp[block_thread_id];
}

template<typename T>
__global__ void gather_sums(T *const array, T const *partial_sums) {
	if (blockIdx.x > 0)
		array[threadIdx.x] += partial_sums[blockIdx.x - 1];
}

template<typename T>
void prefix_recursive(T *const output, T *const input, const std::size_t size, cudaStream_t &stream) {
	const auto num_of_blocks = size / threads_per_block + 1;
	std::size_t pow {1};
	while (pow < threads_per_block)
		pow *= 2;

	GPUBuffer<T> partial_sums {num_of_blocks - 1};
	prescan<<<num_of_blocks, threads_per_block, 2 * threads_per_block, stream>>>
		(output, input, partial_sums.data(), size, pow);
	assert(cudaGetLastError() == cudaSuccess);
	if (num_of_blocks <= 1)
		return;

	auto cuda_status = cudaDeviceSynchronize();
	assert(cuda_status == cudaSuccess);
	prefix_recursive(partial_sums.data(), partial_sums.data(), partial_sums.size(), stream);
	gather_sums<<<num_of_blocks, threads_per_block, 0, stream>>>(output, partial_sums.data());
	assert(cudaGetLastError() == cudaSuccess);
}

template<typename T>
void mapped_scan(GPUBuffer<T> &output, GPUBuffer<T> &input, const std::size_t size,
		 const T target_value, GPUStream &stream) {
	const auto num_of_blocks = size / threads_per_block + 1;
	GPUBuffer<T> mapped_input {size};

	map_to_target<<<num_of_blocks, threads_per_block, 0, stream.raw()>>>
		(mapped_input.data(), input.data(), size, target_value);
	assert(cudaGetLastError() == cudaSuccess);
	prefix_recursive(output.data(), mapped_input.data(), size, stream.raw());
}

} // namespace wf

#endif
