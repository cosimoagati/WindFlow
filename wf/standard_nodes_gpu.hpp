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
 *  @file    standard_nodes_gpu.hpp
 *  @author  Cosimo Agati
 *  @date    12/02/2019
 *
 *  @brief Standard emitter and collector used by the WindFlow library
 *
 *  @section Standard Emitter and Collector (Description)
 *
 *  This file implements the standard emitter and collector used by the library.
 */

#ifndef STANDARD_GPU_H
#define STANDARD_GPU_H

#include <cstddef>
#include <vector>
#include "basic.hpp"
#include <ff/multinode.hpp>
#include "basic_emitter.hpp"

namespace wf {
template<typename tuple_t>
class Standard_EmitterGPU: public Basic_Emitter {
private:
	// type of the function to map the key hashcode onto an identifier starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;
	bool isKeyed; // flag stating whether the key-based distribution is used or not
	routing_func_t routing_func; // routing function
	bool isCombined; // true if this node is used within a Tree_Emitter node
	std::vector<std::pair<void *, int>> output_queue; // used in case of Tree_Emitter mode
	std::size_t dest_w; // used to select the destination
	std::size_t n_dest; // number of destinations


public:
	Standard_EmitterGPU(const std::size_t n_dest)
		: isKeyed {false}, isCombined {false}, dest_w(0), n_dest(n_dest)
	{}

	Standard_EmitterGPU(const routing_func_t routing_func,
			    const std::size_t n_dest)
		: isKeyed {true}, routing_func {routing_func},
		  isCombined {false}, dest_w {0}, n_dest {n_dest}
	{}

	Basic_Emitter *clone() const {
		return new Standard_EmitterGPU<tuple_t>(*this);
	}

	int svc_init() { return 0; }

	void *svc(void *const in) {
		tuple_t *t = reinterpret_cast<tuple_t *>(in);
		if (isKeyed) { // keyed-based distribution enabled
			// extract the key from the input tuple
			auto key = std::get<0>(t->getControlFields()); // key
			auto hashcode = std::hash<decltype(key)>()(key); // compute the hashcode of the key
			// evaluate the routing function
			dest_w = routing_func(hashcode, n_dest);
			// send the tuple
			if (!isCombined) {
				this->ff_send_out_to(t, dest_w);
			} else {
				output_queue.push_back(std::make_pair(t, dest_w));
			}
			return this->GO_ON;
		}
		if (!isCombined) {
			return t;
		}
		output_queue.push_back(std::make_pair(t, dest_w));
		dest_w = (dest_w + 1) % n_dest;
		return this->GO_ON;
	}

	void svc_end() {}

	size_t getNDestinations() const { return n_dest; }

	void setTree_EmitterMode(const bool val) { isCombined = val; }

	std::vector<std::pair<void *, int>> &getOutputQueue() {
		return output_queue;
	}

	// No copying nor moving!
	Standard_EmitterGPU(const Standard_EmitterGPU &) = delete;
	Standard_EmitterGPU(Standard_EmitterGPU &&) = delete;
	Standard_EmitterGPU &operator=(const Standard_EmitterGPU &) = delete;
	Standard_EmitterGPU &operator=(Standard_EmitterGPU &&) = delete;
};

// TODO
// class Standard_Collector
class Standard_CollectorGPU: public ff::ff_minode {
public:
	Standard_CollectorGPU(const ordering_mode_t mode=TS) {
		assert(mode == TS);
	}

	void *svc(void *const t) { return t; }
};

} // namespace wf

#endif
