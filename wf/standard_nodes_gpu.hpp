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
 *  @brief Standard GPU emitter and collector used by the WindFlow library
 *
 *  @section Standard Emitter and Collector (GPU versions) (Description)
 *
 *  This file implements the standard GPU emitter and collector used by the library.
 */

#ifndef STANDARD_GPU_H
#define STANDARD_GPU_H

#include <cstddef>
#include <functional>
#include <vector>
#include "basic.hpp"
#include <ff/multinode.hpp>
#include "basic_emitter.hpp"

namespace wf {
template<typename tuple_t>
class Standard_EmitterGPU: public Basic_Emitter {
private:
	// type of the function to map the key hashcode onto an identifier
	// starting from zero to pardegree-1
	using routing_func_t = std::function<size_t(size_t, size_t)>;
	using key_t = std::remove_reference_t<decltype(std::get<0>(tuple_t {}.getControlFields()))>;

	routing_modes_t routing_mode;
	routing_func_t routing_func; // routing function
	std::hash<key_t> hash;
	std::vector<std::pair<void *, int>> output_queue; // used in case of Tree_Emitter mode
	std::size_t destination_index;
	std::size_t num_of_destinations;
	bool is_combined; // true if this node is used within a Tree_Emitter node
	bool have_gpu_input;
	bool have_gpu_output;
public:
	Standard_EmitterGPU(const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false)
		: routing_mode {FORWARD}, is_combined {false}, destination_index {0},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{}

	Standard_EmitterGPU(const routing_func_t routing_func,
			    const std::size_t num_of_destinations,
			    const bool have_gpu_input=false,
			    const bool have_gpu_output=false)
		: routing_mode {FORWARD}, routing_func {routing_func},
		  is_combined {false}, destination_index {0},
		  num_of_destinations {num_of_destinations},
		  have_gpu_input {have_gpu_input},
		  have_gpu_output {have_gpu_output}
	{}

	// Why does this method exist? Isn't this redundant?
	Basic_Emitter *clone() const {
		return new Standard_EmitterGPU<tuple_t> {*this};
	}

	int svc_init() const { return 0; }

	void *svc(void *const in) {
		tuple_t *t = reinterpret_cast<tuple_t *>(in);
		if (routing_mode == KEYBY && !have_gpu_input) {
			auto key = std::get<0>(t->getControlFields());
			auto hashcode = hash(key);
			destination_index = routing_func(hashcode, num_of_destinations);
			// send the tuple
			if (!is_combined) {
				this->ff_send_out_to(t, destination_index);
			} else {
				output_queue.push_back(std::make_pair(t, destination_index));
			}
			return this->GO_ON;
		}
		if (!is_combined) {
			return t;
		}
		output_queue.push_back(std::make_pair(t, destination_index));
		destination_index = (destination_index + 1) % num_of_destinations;
		return this->GO_ON;
	}

	void svc_end() const {}

	std::size_t getNDestinations() const { return num_of_destinations; }

	void setTree_EmitterMode(const bool val) { is_combined = val; }

	std::vector<std::pair<void *, int>> &getOutputQueue() {
		return output_queue;
	}
};

// class Standard_Collector
// FIXME: Is this needed?
class Standard_CollectorGPU: public ff::ff_minode {
public:
	Standard_CollectorGPU(const ordering_mode_t mode=TS) {
		assert(mode == TS);
	}

	void *svc(void *const t) { return t; }
};

} // namespace wf

#endif
