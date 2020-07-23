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
 *  @brief Various utilities used by MapGPU (to be merged with other utils
 *  files)
 *
 *  @section MapGPU_Node (Description)
 */

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

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
 * Resizes buffer to new_size if it's larger than old_size. the buffer must be
 * allocated on the host via CUDA.
 * Returns true on successful operation, false otherwise.
 */
template<typename T>
inline bool enlarge_cpu_buffer(T *&buffer, const int new_size,
			       const int old_size) {
	if (old_size >= new_size) {
		return true;
	}
	T *tmp;
	if (cudaMallocHost(&tmp, new_size * sizeof *buffer) != cudaSuccess) {
		return false;
	}
	cudaFreeHost(buffer);
	buffer = tmp;
	return true;
}

/*
 * Resizes buffer to new_size if it's larger than old_size. the buffer must be
 * allocated on the device via CUDA.
 * Returns true on successful operation, false otherwise.
 */
template<typename T>
inline bool enlarge_gpu_buffer(T *&buffer, const int new_size,
			       const int old_size) {
	if (old_size >= new_size) {
		return true;
	}
	T *tmp;
	while (cudaMalloc(&tmp, new_size * sizeof *buffer) != cudaSuccess) {
		return false;
	}
	cudaFree(buffer);
	buffer = tmp;
	return true;
}

inline void failwith(const std::string &err) {
	std::cerr << RED << "WindFlow Error: " << err << DEFAULT_COLOR
		  << std::endl;
	std::exit(EXIT_FAILURE);
}
} // namespace wf

#endif
