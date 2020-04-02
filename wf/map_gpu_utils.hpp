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

#ifndef MAP_GPU_UTILS_H
#define MAP_GPU_UTILS_H

#include <functional>
#include <type_traits>
#include <iostream>

#include "basic.hpp"

namespace wf
{
template<typename F, typename... Args>
struct is_invocable :
		std::is_constructible<std::function<void(Args ...)>,
				      std::reference_wrapper<typename std::remove_reference<F>::type>>
{};

inline void
failwith(const std::string &err)
{
	std::cerr << RED << "WindFlow Error: " << err << DEFAULT_COLOR
		  << std::endl;
	std::exit(EXIT_FAILURE);
}
} // namespace wf

#endif
