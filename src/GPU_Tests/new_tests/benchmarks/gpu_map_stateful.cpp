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

/*
 *  Version with { Sources } -> { Map } -> Sink. Stateful Map on GPU.
 */

// includes
#include "../../zipf.hpp"
#include "robin.hpp"
#include <algorithm>
#include <ff/buffer.hpp>
#include <ff/ff.hpp>
#include <ff/mpmc/MPMCqueues.hpp>
#include <iostream>
#include <random>
#include <regex>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace ff;

struct dummy_mi : ff_minode {
	void *svc(void *t) { return t; }
};

// compute the next power of two greater than a 32-bit integer
int32_t next_power_of_two(int32_t n) {
	assert(n > 0);
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	return n + 1;
}

// information contained in each record in the dataset
typedef enum {
	DATE_FIELD,
	TIME_FIELD,
	EPOCH_FIELD,
	DEVICE_ID_FIELD,
	TEMP_FIELD,
	HUMID_FIELD,
	LIGHT_FIELD,
	VOLT_FIELD
} record_field;

// fields that can be monitored by the user
typedef enum { TEMPERATURE, HUMIDITY, LIGHT, VOLTAGE } monitored_field;

// model parameters
size_t          _moving_avg_win_size = 1000;
monitored_field _field               = TEMPERATURE;
double          _threshold           = 0.025;
unsigned long   app_run_time         = 60 * 1000000000L;

struct tuple_t {
	double   property_value;
	double   incremental_average;
	size_t   key;
	uint64_t id;
	uint64_t ts;

	tuple_t() : property_value(0.0), incremental_average(0.0), key(0), id(0), ts(0) {}

	tuple_t(double _property_value, double _incremental_average, size_t _key, uint64_t _id, uint64_t _ts)
	        : property_value(_property_value), incremental_average(_incremental_average), key(_key),
	          id(_id), ts(_ts) {}

	std::tuple<size_t, uint64_t, uint64_t> getControlFields() const {
		return tuple<size_t, uint64_t, uint64_t>(key, id, ts);
	}

	void setControlFields(size_t _key, uint64_t _id, uint64_t _ts) {
		key = _key;
		id  = _id;
		ts  = _ts;
	}
};

// type of the input records: < date_value, time_value, epoch_value, device_id_value, temp_value, humid_value,
// light_value, voltage_value>
using record_t = tuple<string, string, int, int, double, double, double, double>;

// global variables
vector<record_t>                parsed_file;
vector<tuple_t>                 dataset;
unordered_map<size_t, uint64_t> key_occ;
atomic<long>                    sent_tuples;
atomic<long>                    num_allocated_batches;

// function to get the time from the epoch in microseconds
inline unsigned long current_time_usecs() __attribute__((always_inline));
inline unsigned long current_time_usecs() {
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec) * 1000000L + (t.tv_nsec / 1000);
}

// function to get the time from the epoch in nanoseconds
inline unsigned long current_time_nsecs() __attribute__((always_inline));
inline unsigned long current_time_nsecs() {
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec) * 1000000000L + t.tv_nsec;
}

#ifndef NDEBUG
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#define gpuErrChk(ans)                                                                                       \
	do {                                                                                                 \
		gpuAssert((ans), __FILE__, __LINE__);                                                        \
	} while (0)
#else
// Unlike the standard C assert macro, gpuErrChk simply expands to the statement itself if NDEBUG is defined,
// making it suitable to be used with expressions producing side effects.
#define gpuErrChk(ans) (ans)
#endif // NDEBUG

template<typename in_t, typename key_t = std::false_type>
struct batch_t {
	size_t          size;           // size of the batch
	in_t *          data_gpu;       // pointer to the first tuple in the batch
	in_t *          raw_data_gpu;   // pointer to the GPU array shared by this batch
	MPMC_Ptr_Queue *queue;          // pointer to the queue to be used for recycling the GPU array
	atomic<size_t> *delete_counter; // pointer to the atomic counter used to delete the GPU array
	struct keyby_record             // records of fields used for the keyby distribution
	{
		size_t          num_dist_keys;  // number of distinct keys to be read
		key_t *         dist_keys_cpu;  // array of distinct keys to be read
		int *           start_idxs_cpu; // array of starting indexes of keys to be read
		int *           map_idxs_cpu;   // map array to find tuples with the same key in the batch
		atomic<size_t> *ready_counter;  // pointer to the atomic counter to be used to complete the
		                                // keyby processing
	};
	keyby_record kb; // keyby record

	batch_t(size_t _size, in_t *_data_gpu, in_t *_raw_data_gpu, MPMC_Ptr_Queue *_queue,
	        atomic<size_t> *_delete_counter, atomic<size_t> *_ready_counter = nullptr)
	        : size(_size), data_gpu(_data_gpu), raw_data_gpu(_raw_data_gpu), queue(_queue),
	          delete_counter(_delete_counter) {
		kb.ready_counter  = _ready_counter;
		kb.num_dist_keys  = 0;
		kb.dist_keys_cpu  = (key_t *) malloc(sizeof(key_t) * size);
		kb.start_idxs_cpu = (int *) malloc(sizeof(int) * size);
		kb.map_idxs_cpu   = (int *) malloc(sizeof(int) * size);
		std::fill(kb.map_idxs_cpu, kb.map_idxs_cpu + size, -1);
	}

	~batch_t() {
		free(kb.dist_keys_cpu);
		free(kb.start_idxs_cpu);
		free(kb.map_idxs_cpu);
		size_t old_value = delete_counter->fetch_sub(1);
		if (old_value == 1) {
#if __RECYCLE__
			// try to push the GPU array into the recycling queue
			if (!queue->push((void *const) raw_data_gpu))
				gpuErrChk(cudaFree(raw_data_gpu));
#else
			gpuErrChk(cudaFree(raw_data_gpu));
#endif
			delete delete_counter;
		}
	}

	// method to check whether the keyby processing is complete
	bool isKBDone() {
		size_t old_value = (kb.ready_counter)->fetch_sub(1);
		if (old_value == 1) {
			delete kb.ready_counter;
			return true;
		} else {
			return false;
		}
	}
};

class Source : public ff_monode_t<batch_t<tuple_t, size_t>> {
private:
	size_t                     n_dest;
	vector<tuple_t> &          dataset;
	size_t                     next_tuple_idx;
	long                       generated_tuples;
	size_t                     batch_size;
	size_t                     generated_batches;
	size_t                     allocated_batches;
	unsigned long              app_start_time;
	unsigned long              current_time;
	unsigned long              interval;
	size_t                     tuple_id       = 0;
	tuple_t *                  data_cpu[2]    = {nullptr};
	tuple_t *                  data_gpu[2]    = {nullptr};
	batch_t<tuple_t, size_t> **bouts          = nullptr;
	batch_t<tuple_t, size_t> **previous_bouts = nullptr;
	cudaStream_t               cudaStreams[2];
	MPMC_Ptr_Queue *           recycle_queue = nullptr;
	// unordered_map<size_t, size_t> dist_map;
	robin_hood::unordered_map<size_t, size_t> dist_map;
	int                                       id_r = 0;

public:
	Source(size_t _n_dest, vector<tuple_t> &_dataset, size_t _batch_size, unsigned long _app_start_time,
	       MPMC_Ptr_Queue *_recycle_queue)
	        : n_dest(_n_dest), dataset(_dataset), batch_size(_batch_size),
	          app_start_time(_app_start_time), current_time(_app_start_time), next_tuple_idx(0),
	          generated_tuples(0), generated_batches(0), allocated_batches(0),
	          recycle_queue(_recycle_queue) {
		interval = 1000000L;
		// allocate data_cpus in pinned memory
		gpuErrChk(cudaMallocHost(&data_cpu[0], sizeof(tuple_t) * batch_size));
		gpuErrChk(cudaMallocHost(&data_cpu[1], sizeof(tuple_t) * batch_size));
		// initialize CUDA streams
		gpuErrChk(cudaStreamCreate(&cudaStreams[0]));
		gpuErrChk(cudaStreamCreate(&cudaStreams[1]));
		// allocate array of pointers to the batches
		bouts = (batch_t<tuple_t, size_t> **) malloc(sizeof(batch_t<tuple_t, size_t> *) * n_dest);
		previous_bouts =
		        (batch_t<tuple_t, size_t> **) malloc(sizeof(batch_t<tuple_t, size_t> *) * n_dest);
	}

	~Source() {
		// deallocate data_cpus from pinned memory
		gpuErrChk(cudaFreeHost(data_cpu[0]));
		gpuErrChk(cudaFreeHost(data_cpu[1]));
		// deallocate CUDA streams
		gpuErrChk(cudaStreamDestroy(cudaStreams[0]));
		gpuErrChk(cudaStreamDestroy(cudaStreams[1]));
		// deallocate array of pointers to the batches
		free(bouts);
		free(previous_bouts);
	}

	batch_t<tuple_t, size_t> *svc(batch_t<tuple_t, size_t> *) {
		bool endGeneration = false;
		while (!endGeneration) {
			if (generated_tuples > 0)
				current_time = current_time_nsecs();
			generated_tuples++;
			tuple_t t;
			// prepare the tuple by reading the dataset
			tuple_t tuple         = dataset.at(next_tuple_idx);
			t.property_value      = tuple.property_value;
			t.incremental_average = tuple.incremental_average;
			t.key                 = tuple.key;
			t.id                  = tuple.id;
			t.ts                  = current_time - app_start_time;
			next_tuple_idx        = (next_tuple_idx + 1) % dataset.size();
			// check EOS
			if (current_time - app_start_time >= app_run_time && next_tuple_idx == 0) {
				sent_tuples.fetch_add(generated_tuples);
				num_allocated_batches.fetch_add(allocated_batches);
				endGeneration = true;
			}
			prepare_batch(t);
		}
		return this->EOS;
	}

	void prepare_batch(tuple_t &t) {
		if (data_gpu[id_r] == nullptr) {
#ifdef __RECYCLE__
			// try to recycle previously allocated GPU array
			if (!recycle_queue->pop((void **) &(data_gpu[id_r]))) {
				gpuErrChk(cudaMalloc(&(data_gpu[id_r]), sizeof(tuple_t) * batch_size));
				allocated_batches++;
			}
#else
			gpuErrChk(cudaMalloc(&(data_gpu[id_r]), sizeof(tuple_t) * batch_size));
			allocated_batches++;
#endif
			// allocate batches to be sent
			atomic<size_t> *delete_counter = new atomic<size_t>(n_dest);
			atomic<size_t> *ready_counter  = new atomic<size_t>(n_dest);
			for (size_t i = 0; i < n_dest; i++) {
				bouts[i] = new batch_t<tuple_t, size_t>(batch_size, data_gpu[id_r],
				                                        data_gpu[id_r], recycle_queue,
				                                        delete_counter, ready_counter);
			}
		}
		// copy the input tuple in the pinned buffer
		data_cpu[id_r][tuple_id] = t;
		// copy the key attribute of the input tuple in the pinned buffer in the batch
		auto key = std::get<0>(t.getControlFields());
		// prepare the distribution
		const auto id_dest = (key % n_dest);
		auto       it      = dist_map.find(key);
		auto &     kb      = bouts[id_dest]->kb;
		if (it == dist_map.end()) {
			// dist_map.insert(std::make_pair(key, tuple_id));
			dist_map.insert(robin_hood::pair<size_t, size_t>(key, tuple_id));
			kb.dist_keys_cpu[kb.num_dist_keys]  = key;
			kb.start_idxs_cpu[kb.num_dist_keys] = tuple_id;
			kb.num_dist_keys++;
		} else {
			kb.map_idxs_cpu[it->second] = tuple_id;
			it->second                  = tuple_id;
		}
		tuple_id++;
		if (tuple_id == batch_size) {
			if (generated_batches > 0) {
				gpuErrChk(cudaStreamSynchronize(cudaStreams[(id_r + 1) % 2]));
				for (size_t i = 0; i < n_dest; i++)
					this->ff_send_out_to(previous_bouts[i], i);
			}
			generated_batches++;
			// copy the tuples in the GPU area
			gpuErrChk(cudaMemcpyAsync(data_gpu[id_r], data_cpu[id_r],
			                          batch_size * sizeof(tuple_t), cudaMemcpyHostToDevice,
			                          cudaStreams[id_r]));
			data_gpu[id_r] = nullptr;
			for (size_t i = 0; i < n_dest; i++) {
				previous_bouts[i] = bouts[i];
				bouts[i]          = nullptr;
			}
			tuple_id = 0;
			dist_map.clear();
			id_r = (id_r + 1) % 2;
		}
	}

	void eosnotify(ssize_t id) {
		if (generated_batches > 0) {
			gpuErrChk(cudaStreamSynchronize(cudaStreams[(id_r + 1) % 2]));
			for (size_t i = 0; i < n_dest; i++)
				this->ff_send_out_to(previous_bouts[i], i);
		}
	}

	void svc_end() {
		bool     end = false;
		tuple_t *ptr = nullptr;
		while (!end) {
			if (recycle_queue->pop((void **) &ptr))
				gpuErrChk(cudaFree(ptr));
			else
				end = true;
		}
	}
};

struct Window_State {
	double values[1000];
	// double *values=nullptr;
	double sum;
	size_t first;
	size_t last;
	size_t count;

	__device__ Window_State() {
		// values = (double *) malloc(1000 * sizeof(double));
		// assert(values != nullptr); // malloc in device code can fail!
		sum   = 0;
		first = 0;
		last  = 0;
		count = 0;
	}

	__device__ double compute(double next_value) {
		if (count == 1000) {
			sum -= values[first];
			first = (first + 1) % 1000;
			sum += next_value;
			values[last] = next_value;
			last         = (last + 1) % 1000;
		} else {
			values[last] = next_value;
			last         = (last + 1) % 1000;
			sum += next_value;
			count++;
		}
		return sum / count;
	}
};

// function to be executed on each item in a batch
__device__ void map_function(tuple_t &t, Window_State &state) {
	t.incremental_average = state.compute(t.property_value);
}

#if !defined(__SHARED__)
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, int *map_idxs, int *start_idxs,
                                           Window_State **states, int num_dist_keys,
                                           int num_active_thread_per_warp) {
	const int thread_id   = threadIdx.x + blockIdx.x * blockDim.x;
	const int num_threads = gridDim.x * blockDim.x;

	// A "worker" is a thread that is actually doing work.
	const int threads_per_worker = warpSize / num_active_thread_per_warp;
	const int num_workers        = num_threads / threads_per_worker;
	const int worker_id          = thread_id / threads_per_worker;
	// only one thread each threads_per_worker threads works, the others are idle
	if (thread_id % threads_per_worker == 0) {
		for (int key_id = worker_id; key_id < num_dist_keys; key_id += num_workers) {
			size_t idx = start_idxs[key_id];
			// execute all the inputs with key in the input batch
			while (idx != -1) {
				map_function(tuples[idx], *(states[key_id]));
				idx = map_idxs[idx];
			}
		}
	}
}
#else
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, int *map_idxs, int *start_idxs,
                                           Window_State **states, int num_dist_keys,
                                           int num_active_thread_per_warp) {
	extern __shared__ char array[];

	const int  thread_id          = threadIdx.x + blockIdx.x * blockDim.x;
	const int  num_threads        = gridDim.x * blockDim.x;
	const int  threads_per_worker = warpSize / num_active_thread_per_warp;
	const int  num_workers        = num_threads / threads_per_worker;
	const int  worker_id          = thread_id / threads_per_worker;
	const auto cached_states      = reinterpret_cast<Window_State *>(array);

	// only the first thread of each warp works, the others are idle
	if (thread_id % threads_per_worker == 0) {
		auto &cached_state = cached_states[threadIdx.x / threads_per_worker];
		for (int key_id = worker_id; key_id < num_dist_keys; key_id += num_workers) {
			size_t idx   = start_idxs[key_id];
			cached_state = *(states[key_id]);
			// execute all the inputs with key in the input batch
			while (idx != -1) {
				map_function(tuples[idx], cached_state);
				idx = map_idxs[idx];
			}
			*(states[key_id]) = cached_state;
		}
	}
}
#endif

// CUDA Kernel to initialize the states of new keys
__global__ void Initialize_States_Kernel(Window_State **new_states, size_t num_states) {
	const int id          = threadIdx.x + blockIdx.x * blockDim.x;
	const int num_threads = gridDim.x * blockDim.x;
	for (size_t i = id; i < num_states; i += num_threads)
		new (new_states[i]) Window_State();
}

class Map : public ff_node_t<batch_t<tuple_t, size_t>> {
private:
	size_t        id_map;
	size_t        map_degree;
	size_t        processed;
	unsigned long received_batch = 0;
	// unordered_map<size_t, Window_State*> hashmap;
	robin_hood::unordered_map<size_t, Window_State *> hashmap;
	unsigned long                                     app_start_time;
	unsigned long                                     current_time;
	unsigned long                                     tot_elapsed_nsec   = 0;
	int                                               numSMs             = 0;
	int                                               max_threads_per_sm = 0;
	int                                               max_blocks_per_sm  = 0;
	int                                               threads_per_warp   = 0;
	int                                               num_keys_per_batch = 0;
	batch_t<tuple_t, size_t> *                        batch_to_be_sent   = nullptr;
	int                                               eos_received       = 0;
	size_t                                            max_batch_len      = 0;
	struct record_t {
		Window_State **state_ptrs_cpu     = nullptr;
		Window_State **state_ptrs_gpu     = nullptr;
		Window_State **new_state_ptrs_cpu = nullptr;
		Window_State **new_state_ptrs_gpu = nullptr;
		size_t *       dist_keys_gpu      = nullptr;
		int *          start_idxs_gpu     = nullptr;
		int *          map_idxs_gpu       = nullptr;
		cudaStream_t   cudaStream;

		record_t(size_t _size) {
			// initialize CUDA stream
			gpuErrChk(cudaStreamCreate(&cudaStream));
			// create arrays on GPU
			gpuErrChk(cudaMalloc(&state_ptrs_gpu, _size * sizeof(Window_State *)));
			gpuErrChk(cudaMalloc(&new_state_ptrs_gpu, _size * sizeof(Window_State *)));
			gpuErrChk(cudaMalloc(&dist_keys_gpu, _size * sizeof(size_t)));
			gpuErrChk(cudaMalloc(&start_idxs_gpu, _size * sizeof(int)));
			gpuErrChk(cudaMalloc(&map_idxs_gpu, _size * sizeof(int)));
			// create arrays on pinned memory
			gpuErrChk(cudaMallocHost(&state_ptrs_cpu, _size * sizeof(Window_State *)));
			gpuErrChk(cudaMallocHost(&new_state_ptrs_cpu, _size * sizeof(Window_State *)));
		}

		~record_t() {
			// deallocate arrays from GPU
			gpuErrChk(cudaFree(state_ptrs_gpu));
			gpuErrChk(cudaFree(new_state_ptrs_gpu));
			gpuErrChk(cudaFree(dist_keys_gpu));
			gpuErrChk(cudaFree(start_idxs_gpu));
			gpuErrChk(cudaFree(map_idxs_gpu));
			// deallocate arrays from pinned memory
			gpuErrChk(cudaFreeHost(state_ptrs_cpu));
			gpuErrChk(cudaFreeHost(new_state_ptrs_cpu));
			// deallocate CUDA stream
			gpuErrChk(cudaStreamDestroy(cudaStream));
		}
	};
	record_t *records[2];
	size_t    id_r = 0;

public:
	Map(size_t _id_map, size_t _map_degree, const unsigned long _app_start_time, size_t _max_batch_len)
	        : id_map(_id_map), map_degree(_map_degree), processed(0), app_start_time(_app_start_time),
	          current_time(_app_start_time), max_batch_len(_max_batch_len) {
		// get some configuration parameters of the GPU
		gpuErrChk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
		gpuErrChk(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
		                                 0));
#ifdef __aarch64__
		max_blocks_per_sm = 32;

#else
		gpuErrChk(
		        cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
#endif // __aarch64__
		gpuErrChk(cudaDeviceGetAttribute(&threads_per_warp, cudaDevAttrWarpSize, 0));
		assert(numSMs > 0);             // 1
		assert(max_threads_per_sm > 0); //  2048
		assert(max_blocks_per_sm > 0);  // 16
		assert(threads_per_warp > 0);   // 32
	}

	~Map() {
		if (received_batch > 0) {
			delete records[0];
			delete records[1];
		}
	}

	batch_t<tuple_t, size_t> *svc(batch_t<tuple_t, size_t> *b) {
		volatile unsigned long start_time_nsec = current_time_nsecs();
		received_batch++;
		processed += b->size;
		// create the two records (one time only)
		if (received_batch == 1) {
			records[0] = new record_t(max_batch_len);
			records[1] = new record_t(max_batch_len);
		}
		int num_new_keys = 0;
		// create the array of pointers to the state for each unique key
		for (size_t i = 0; i < (b->kb).num_dist_keys; i++) {
			const auto key = (b->kb).dist_keys_cpu[i];
			auto       it  = hashmap.find(key);
			if (it == hashmap.end()) {
				// allocate the memory for the new state on GPU
				Window_State *state_gpu = nullptr;
				gpuErrChk(cudaMalloc(&state_gpu, sizeof(Window_State)));
				records[id_r]->new_state_ptrs_cpu[num_new_keys] = state_gpu;
				num_new_keys++;
				// hashmap.insert(std::make_pair(key, state_gpu));
				hashmap.insert(robin_hood::pair<size_t, Window_State *>(key, state_gpu));
				it = hashmap.find(key);
			}
			records[id_r]->state_ptrs_cpu[i] = (*it).second;
		}
		// initialize new allocated states (if any)
		if (num_new_keys > 0) {
			int threads_per_block = 128;
			int num_blocks = std::min((int) ceil(((double) num_new_keys) / threads_per_block),
			                          numSMs * max_blocks_per_sm);
			Initialize_States_Kernel<<<num_blocks, threads_per_block, 0,
			                           records[id_r]->cudaStream>>>(
			        records[id_r]->new_state_ptrs_cpu, num_new_keys);
		}
		gpuErrChk(cudaMemcpyAsync(records[id_r]->state_ptrs_gpu, records[id_r]->state_ptrs_cpu,
		                          (b->kb).num_dist_keys * sizeof(Window_State *),
		                          cudaMemcpyHostToDevice, records[id_r]->cudaStream));
		gpuErrChk(cudaMemcpyAsync(records[id_r]->dist_keys_gpu, (b->kb).dist_keys_cpu,
		                          (b->kb).num_dist_keys * sizeof(size_t), cudaMemcpyHostToDevice,
		                          records[id_r]->cudaStream));
		gpuErrChk(cudaMemcpyAsync(records[id_r]->start_idxs_gpu, (b->kb).start_idxs_cpu,
		                          (b->kb).num_dist_keys * sizeof(int), cudaMemcpyHostToDevice,
		                          records[id_r]->cudaStream));
		gpuErrChk(cudaMemcpyAsync(records[id_r]->map_idxs_gpu, (b->kb).map_idxs_cpu,
		                          b->size * sizeof(int), cudaMemcpyHostToDevice,
		                          records[id_r]->cudaStream));
		num_keys_per_batch += (b->kb).num_dist_keys;
		// launch the kernel to compute the results
		int warps_per_block = ((max_threads_per_sm / max_blocks_per_sm) / threads_per_warp);
		int tot_num_warps   = warps_per_block * max_blocks_per_sm * numSMs;
		// compute how many threads should be active per warps
		int32_t x = (int32_t) ceil(((double) (b->kb).num_dist_keys) / tot_num_warps);
		if (x > 1)
			x = next_power_of_two(x);
		int num_active_thread_per_warp = std::min(x, threads_per_warp);
		int num_blocks = std::min((int) ceil(((double) (b->kb).num_dist_keys) / warps_per_block),
		                          numSMs * max_blocks_per_sm);
		if (batch_to_be_sent != nullptr) {
			gpuErrChk(cudaStreamSynchronize(records[(id_r + 1) % 2]->cudaStream));
			if (batch_to_be_sent->isKBDone())
				this->ff_send_out(batch_to_be_sent);
			else
				delete batch_to_be_sent;
		}
#if !defined(__SHARED__)
		Stateful_Processing_Kernel<<<num_blocks, warps_per_block * threads_per_warp, 0,
		                             records[id_r]->cudaStream>>>(
		        b->data_gpu, records[id_r]->map_idxs_gpu, records[id_r]->start_idxs_gpu,
		        records[id_r]->state_ptrs_gpu, (b->kb).num_dist_keys, num_active_thread_per_warp);
#else
		Stateful_Processing_Kernel<<<num_blocks, warps_per_block * threads_per_warp,
		                             sizeof(Window_State) * num_active_thread_per_warp
		                                     * warps_per_block,
		                             records[id_r]->cudaStream>>>(
		        b->data_gpu, records[id_r]->map_idxs_gpu, records[id_r]->start_idxs_gpu,
		        records[id_r]->state_ptrs_gpu, (b->kb).num_dist_keys, num_active_thread_per_warp);
#endif
		// Stateful_Processing_Kernel<<<num_blocks, warps_per_block*threads_per_warp,
		// sizeof(Window_State) * num_active_thread_per_warp * warps_per_block,
		// records[id_r]->cudaStream>>>(b->data_gpu, records[id_r]->map_idxs_gpu,
		// records[id_r]->dist_keys_gpu, records[id_r]->start_idxs_gpu, records[id_r]->state_ptrs_gpu,
		// (b->kb).num_dist_keys, num_active_thread_per_warp);
		batch_to_be_sent                         = b;
		id_r                                     = (id_r + 1) % 2;
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return this->GO_ON;
	}

	void eosnotify(ssize_t id) {
		if (batch_to_be_sent != nullptr) {
			gpuErrChk(cudaStreamSynchronize(records[(id_r + 1) % 2]->cudaStream));
			if (batch_to_be_sent->isKBDone())
				this->ff_send_out(batch_to_be_sent);
			else
				delete batch_to_be_sent;
		}
	}

	void svc_end() {
		printf("[MAP] average service time: %f usec\n",
		       (((double) tot_elapsed_nsec) / received_batch) / 1000);
		printf("[MAP] average number of keys per batch: %f\n",
		       ((double) num_keys_per_batch) / received_batch);
	}
};

class Sink : public ff_minode_t<batch_t<tuple_t, size_t>> {
private:
	uint64_t     received;
	uint64_t     received_batches;
	cudaStream_t cudaStream;

public:
	Sink() : received(0), received_batches(0) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	~Sink() {
		// deallocate CUDA stream
		gpuErrChk(cudaStreamDestroy(cudaStream));
	}

	batch_t<tuple_t, size_t> *svc(batch_t<tuple_t, size_t> *b) {
		received_batches++;
#ifndef NDEBUG
		if (received < 100) {
			tuple_t *data_cpu;
			gpuErrChk(cudaMallocHost(&data_cpu, sizeof(tuple_t) * b->size));
			gpuErrChk(cudaMemcpyAsync(data_cpu, b->data_gpu, b->size * sizeof(tuple_t),
			                          cudaMemcpyDeviceToHost, cudaStream));
			gpuErrChk(cudaStreamSynchronize(cudaStream));
			for (size_t i = 0; i < b->size; i++) {
				tuple_t *t = &(data_cpu[i]);
				cout << "Tuple: " << t->key << " " << t->property_value << " "
				     << t->incremental_average << endl;
				if (received + i >= 100)
					break;
			}
			gpuErrChk(cudaFreeHost(data_cpu));
		}
#endif
		received += b->size;
		delete b;
		return this->GO_ON;
	}

	void svc_end() { cout << "[SINK] received " << received << " inputs" << endl; }
};

void parse_dataset(const string &file_path) {
	ifstream file(file_path);
	if (!file.is_open()) {
		cerr << "Error while reading file " << file_path << ", does it exist?" << endl;
		exit(EXIT_FAILURE);
	}
	size_t all_records        = 0;
	size_t incomplete_records = 0;
	string line;
	while (getline(file, line)) {
		// process file line
		int                   token_count = 0;
		vector<string>        tokens;
		regex                 rgx("\\s+"); // regex quantifier (matches one or many whitespaces)
		sregex_token_iterator iter(line.begin(), line.end(), rgx, -1);
		sregex_token_iterator end;
		while (iter != end) {
			tokens.push_back(*iter);
			token_count++;
			iter++;
		}
		// a record is valid if it contains at least 8 values (one for each field of interest)
		if (token_count >= 8) {
			// save parsed file
			record_t r(tokens.at(DATE_FIELD), tokens.at(TIME_FIELD),
			           atoi(tokens.at(EPOCH_FIELD).c_str()),
			           atoi(tokens.at(DEVICE_ID_FIELD).c_str()),
			           atof(tokens.at(TEMP_FIELD).c_str()), atof(tokens.at(HUMID_FIELD).c_str()),
			           atof(tokens.at(LIGHT_FIELD).c_str()), atof(tokens.at(VOLT_FIELD).c_str()));
			parsed_file.push_back(r);
			// insert the key device_id in the map (if it is not present)
			if (key_occ.find(get<DEVICE_ID_FIELD>(r)) == key_occ.end()) {
				key_occ.insert(make_pair(get<DEVICE_ID_FIELD>(r), 0));
			}
		} else {
			incomplete_records++;
		}
		all_records++;
	}
	file.close();
}

void create_tuples(int num_keys) {
	// std::uniform_int_distribution<std::mt19937::result_type> dist(0, num_keys - 1);
	shifted_zipf_distribution<std::mt19937::result_type> dist {0, num_keys - 1};
	mt19937                                              rng;
	rng.seed(0);
	for (int next_tuple_idx = 0; next_tuple_idx < parsed_file.size(); next_tuple_idx++) {
		// create tuple
		auto    record = parsed_file.at(next_tuple_idx);
		tuple_t t;
		// select the value of the field the user chose to monitor (parameter set in constants.hpp)
		if (_field == TEMPERATURE)
			t.property_value = get<TEMP_FIELD>(record);
		else if (_field == HUMIDITY)
			t.property_value = get<HUMID_FIELD>(record);
		else if (_field == LIGHT)
			t.property_value = get<LIGHT_FIELD>(record);
		else if (_field == VOLTAGE)
			t.property_value = get<VOLT_FIELD>(record);
		t.incremental_average = 0;
		if (num_keys > 0)
			t.key = dist(rng);
		else
			t.key = get<DEVICE_ID_FIELD>(record);
		t.id = (key_occ.find(get<DEVICE_ID_FIELD>(record)))->second++;
		t.ts = 0L;
		dataset.insert(dataset.end(), t);
	}
}

int main(int argc, char *argv[]) {
	const auto arg_error_message = string {argv[0]}
	                               + " -s [num sources] -k [num keys] -b [batch length] "
	                                 "-n [map degree] -f [input file]";
	/// parse arguments from command line
	int    option = 0;
	int    index  = 0;
	string file_path;
	sent_tuples           = 0;
	num_allocated_batches = 0;
	size_t batch_size     = 1;
	int    num_keys       = 0;
	int    map_degree     = 0;
	int    num_sources    = 0;
	string input_file     = "";
	// arguments from command line
	if (argc != 11) {
		cout << arg_error_message << endl;

		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "s:k:b:n:f:")) != -1) {
		switch (option) {
		case 's':
			num_sources = atoi(optarg);
			break;
		case 'k':
			num_keys = atoi(optarg);
			break;
		case 'b':
			batch_size = atoi(optarg);
			break;
		case 'n':
			map_degree = atoi(optarg);
			break;
		case 'f':
			input_file = optarg;
			break;
		default: {
			cout << arg_error_message << endl;
			exit(EXIT_SUCCESS);
		}
		}
	}
	// data pre-processing
	parse_dataset(input_file);
	create_tuples(num_keys);
	// application starting time
	unsigned long     app_start_time = current_time_nsecs();
	ff_pipeline *     pipe           = new ff_pipeline();
	ff_a2a *          a2a            = new ff_a2a();
	vector<ff_node *> first_set;
	for (size_t i = 0; i < num_sources; i++) {
		ff_pipeline *   pipe_in = new ff_pipeline();
		MPMC_Ptr_Queue *queue   = new MPMC_Ptr_Queue();
		queue->init(DEFAULT_BUFFER_CAPACITY);
		pipe_in->add_stage(new Source(map_degree, dataset, batch_size, app_start_time, queue), true);
		first_set.push_back(pipe_in);
	}
	a2a->add_firstset(first_set, 0, true);
	vector<ff_node *> second_set;
	for (size_t i = 0; i < map_degree; i++) {
		ff_comb *comb_node = new ff_comb(
		        new dummy_mi(), new Map(i, map_degree, app_start_time, batch_size), true, true);
		second_set.push_back(comb_node);
	}
	a2a->add_secondset(second_set, true);
	pipe->add_stage(a2a, true);
	pipe->add_stage(new Sink(), true);
	cout << "Starting pipe with " << pipe->cardinality() << " threads..." << endl;
	// evaluate topology execution time
	volatile unsigned long start_time_main_usecs = current_time_usecs();
	pipe->run_and_wait_end();
	volatile unsigned long end_time_main_usecs = current_time_usecs();
	double elapsed_time_seconds = (end_time_main_usecs - start_time_main_usecs) / (1000000.0);
	double throughput           = sent_tuples / elapsed_time_seconds;
	cout << "Measured throughput: " << (int) throughput << " tuples/second" << endl;
	cout << "Allocated batches: " << (int) num_allocated_batches << endl;
	cout << "...end" << endl;
	return 0;
}
