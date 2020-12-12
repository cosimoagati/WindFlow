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
 *  Version with { Sources } -> { Filter } -> Sink. Stateless Filter on GPU.
 */
#include "custom_allocator.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <ff/mpmc/MPMCqueues.hpp>
#include <iostream>
#include <random>
#include <regex>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
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

	__host__ __device__ tuple_t() : property_value(0.0), incremental_average(0.0), key(0), id(0), ts(0) {}

	__host__ __device__ tuple_t(double _property_value, double _incremental_average, size_t _key,
	                            uint64_t _id, uint64_t _ts)
	        : property_value(_property_value), incremental_average(_incremental_average), key(_key),
	          id(_id), ts(_ts) {}

	__host__ __device__ std::tuple<size_t, uint64_t, uint64_t> getControlFields() const {
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

// assert function on GPU
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
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
				cudaFree(raw_data_gpu);
#else
			cudaFree(raw_data_gpu);
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
	size_t                    n_dest;
	vector<tuple_t> &         dataset;
	size_t                    next_tuple_idx;
	long                      generated_tuples;
	size_t                    batch_size;
	size_t                    generated_batches;
	size_t                    allocated_batches;
	unsigned long             app_start_time;
	unsigned long             current_time;
	unsigned long             interval;
	size_t                    tuple_id         = 0;
	tuple_t *                 data_cpu[2]      = {nullptr};
	tuple_t *                 data_gpu[2]      = {nullptr};
	batch_t<tuple_t, size_t> *b                = nullptr;
	batch_t<tuple_t, size_t> *batch_to_be_sent = nullptr;
	cudaStream_t              cudaStreams[2];
	MPMC_Ptr_Queue *          recycle_queue = nullptr;
	int                       id_r          = 0;

public:
	Source(size_t _n_dest, vector<tuple_t> &_dataset, size_t _batch_size, unsigned long _app_start_time,
	       MPMC_Ptr_Queue *_recycle_queue)
	        : n_dest(_n_dest), dataset(_dataset), batch_size(_batch_size),
	          app_start_time(_app_start_time), current_time(_app_start_time), next_tuple_idx(0),
	          generated_tuples(0), generated_batches(0), allocated_batches(0),
	          recycle_queue(_recycle_queue) {
		interval = 1000000L;
		// allocate data_cpus in pinned memory
		cudaMallocHost(&data_cpu[0], sizeof(tuple_t) * batch_size);
		cudaMallocHost(&data_cpu[1], sizeof(tuple_t) * batch_size);
		// initialize CUDA streams
		gpuErrChk(cudaStreamCreate(&cudaStreams[0]));
		gpuErrChk(cudaStreamCreate(&cudaStreams[1]));
	}

	~Source() {
		// deallocate data_cpus from pinned memory
		cudaFreeHost(data_cpu[0]);
		cudaFreeHost(data_cpu[1]);
		// deallocate CUDA streams
		gpuErrChk(cudaStreamDestroy(cudaStreams[0]));
		gpuErrChk(cudaStreamDestroy(cudaStreams[1]));
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
				cudaMalloc(&(data_gpu[id_r]), sizeof(tuple_t) * batch_size);
				allocated_batches++;
			}
#else
			cudaMalloc(&(data_gpu[id_r]), sizeof(tuple_t) * batch_size);
			allocated_batches++;
#endif
			// allocate the new batch
			atomic<size_t> *delete_counter = new atomic<size_t>(1);
			b = new batch_t<tuple_t, size_t>(batch_size, data_gpu[id_r], data_gpu[id_r],
			                                 recycle_queue, delete_counter);
		}
		// copy the input tuple in the pinned buffer
		data_cpu[id_r][tuple_id] = t;
		tuple_id++;
		if (tuple_id == batch_size) {
			if (generated_batches > 0) {
				gpuErrChk(cudaStreamSynchronize(cudaStreams[(id_r + 1) % 2]));
				this->ff_send_out(batch_to_be_sent);
			}
			generated_batches++;
			// copy the tuples in the GPU area
			gpuErrChk(cudaMemcpyAsync(data_gpu[id_r], data_cpu[id_r],
			                          batch_size * sizeof(tuple_t), cudaMemcpyHostToDevice,
			                          cudaStreams[id_r]));
			data_gpu[id_r]   = nullptr;
			batch_to_be_sent = b;
			b                = nullptr;
			tuple_id         = 0;
			id_r             = (id_r + 1) % 2;
		}
	}

	void eosnotify(ssize_t id) {
		if (generated_batches > 0) {
			gpuErrChk(cudaStreamSynchronize(cudaStreams[(id_r + 1) % 2]));
			this->ff_send_out(batch_to_be_sent);
		}
	}

	void svc_end() {
		bool     end = false;
		tuple_t *ptr = nullptr;
		while (!end) {
			if (recycle_queue->pop((void **) &ptr))
				cudaFree(ptr);
			else
				end = true;
		}
	}
};

// struct element of the hashtable
struct node_t {
	size_t key;
	size_t value;
};

class Filter_Functor {
private:
	int     size;
	int     mask;
	node_t *table_cpu;
	node_t *table_gpu;

public:
	Filter_Functor(size_t _n_keys) : size(_n_keys), mask(_n_keys - 1) {
		// create the table on cpu
		table_cpu = (node_t *) malloc(sizeof(node_t) * size);
		// create random number generator
		std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
		mt19937                                                  rng;
		rng.seed(0);
		// populate the table
		for (size_t i = 0; i < size; i++) {
			table_cpu[i].key   = i;
			table_cpu[i].value = dist(rng);
		}
		// copy the table on gpu
		cudaMalloc(&table_gpu, sizeof(node_t) * size);
		cudaMemcpy(table_gpu, table_cpu, sizeof(node_t) * size, cudaMemcpyHostToDevice);
	}

	~Filter_Functor() {
		// free(table_cpu);
		// cudaFree(table_gpu);
	}

	__device__ size_t get_value(const size_t key) {
		int ind = key & mask;
		int i   = ind;
		for (; i < size; i++) {
			if (table_gpu[i].key == key)
				return table_gpu[i].value;
		}
		for (i = 0; i < ind; i++) {
			if (table_gpu[i].key == key)
				return table_gpu[i].value;
		}
		return 0;
	}

	__device__ bool operator()(tuple_t &t) {
		size_t value = get_value(std::get<0>(t.getControlFields()));
		t.id         = value;
		return (value < 100);
	}
};

__global__ void Stateless_Processing_Kernel(tuple_t *tuples, bool *flags, size_t len, Filter_Functor &_func,
                                            int num_active_thread_per_warp) {
	int id          = threadIdx.x + blockIdx.x * blockDim.x; // id of the thread in the kernel
	int num_threads = gridDim.x * blockDim.x;                // number of threads in the kernel
	int threads_per_worker =
	        warpSize / num_active_thread_per_warp;      // number of threads composing a worker entity
	int num_workers = num_threads / threads_per_worker; // number of workers
	int id_worker   = id / threads_per_worker;          // id of the worker corresponding to this thread
	// only "num_active_thread_per_warp" threads per warp work, the others are idle
	if (id % threads_per_worker == 0) {
		for (size_t i = id; i < len; i += num_threads)
			flags[i] = _func(tuples[i]);
	}
}

class Filter : public ff_node_t<batch_t<tuple_t, size_t>> {
private:
	size_t                    id_map;
	size_t                    map_degree;
	size_t                    processed;
	unsigned long             received_batch = 0;
	unsigned long             app_start_time;
	unsigned long             current_time;
	unsigned long             tot_elapsed_nsec   = 0;
	int                       numSMs             = 0;
	int                       max_threads_per_sm = 0;
	int                       max_blocks_per_sm  = 0;
	int                       threads_per_warp   = 0;
	int                       num_keys_per_batch = 0;
	batch_t<tuple_t, size_t> *batch_to_be_sent   = nullptr;
	int                       eos_received       = 0;
	size_t                    max_batch_len      = 0;
	cudaStream_t              cudaStream;
	Filter_Functor            filterF;
	Filter_Functor *          filterF_gpu;
	bool *                    flags_gpu    = nullptr;
	tuple_t *                 new_data_gpu = nullptr;
	cached_allocator          alloc;

public:
	Filter(size_t _id_map, size_t _map_degree, const unsigned long _app_start_time, size_t _max_batch_len,
	       size_t _num_keys)
	        : id_map(_id_map), map_degree(_map_degree), processed(0), app_start_time(_app_start_time),
	          current_time(_app_start_time), max_batch_len(_max_batch_len), filterF(_num_keys) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
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
		cudaMalloc(&flags_gpu, sizeof(bool) * max_batch_len);
		cudaMalloc(&new_data_gpu, sizeof(tuple_t) * max_batch_len);
		// allocate the functor object on GPU
		cudaMalloc(&filterF_gpu, sizeof(Filter_Functor));
		// copy the functor object on GPU
		gpuErrChk(cudaMemcpyAsync(filterF_gpu, &filterF, sizeof(Filter_Functor),
		                          cudaMemcpyHostToDevice, cudaStream));
	}

	~Filter() {
		gpuErrChk(cudaStreamDestroy(cudaStream));
		cudaFree(flags_gpu);
		cudaFree(new_data_gpu);
	}

	batch_t<tuple_t, size_t> *svc(batch_t<tuple_t, size_t> *b) {
		volatile unsigned long start_time_nsec = current_time_nsecs();
		received_batch++;
		processed += b->size;
		// proces the batch in a stateless manner
		int warps_per_block = ((max_threads_per_sm / max_blocks_per_sm) / threads_per_warp);
		int tot_num_warps   = warps_per_block * max_blocks_per_sm * numSMs;
		// compute how many threads should be active per warps
		int32_t x = (int32_t) ceil(((double) (b->size)) / tot_num_warps);
		if (x > 1)
			x = next_power_of_two(x);
		int num_active_thread_per_warp = std::min(x, threads_per_warp);
		int num_blocks                 = std::min((int) ceil(((double) (b->size)) / warps_per_block),
                                          numSMs * max_blocks_per_sm);
		Stateless_Processing_Kernel<<<num_blocks, warps_per_block * threads_per_warp, 0,
		                              cudaStream>>>(b->data_gpu, flags_gpu, b->size, *filterF_gpu,
		                                            num_active_thread_per_warp);
		// compact the output batch
		thrust::device_ptr<bool>    th_flags_gpu    = thrust::device_pointer_cast(flags_gpu);
		thrust::device_ptr<tuple_t> th_data_gpu     = thrust::device_pointer_cast(b->data_gpu);
		thrust::device_ptr<tuple_t> th_new_data_gpu = thrust::device_pointer_cast(new_data_gpu);
		auto                        pred            = [] __device__(bool x) { return x; };
		auto end = thrust::copy_if(thrust::cuda::par(alloc).on(cudaStream), th_data_gpu,
		                           th_data_gpu + b->size, flags_gpu, th_new_data_gpu, pred);
		// change the logical size of the batch
		b->size = end - th_new_data_gpu;
		gpuErrChk(cudaMemcpyAsync(b->data_gpu, new_data_gpu, b->size * sizeof(tuple_t),
		                          cudaMemcpyDeviceToDevice, cudaStream));
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return b;
	}

	void svc_end() {
		printf("[FILTER] average service time: %f usec\n",
		       (((double) tot_elapsed_nsec) / received_batch) / 1000);
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
#if 0
        if (received < 100) {
            tuple_t *data_cpu;
            cudaMallocHost(&data_cpu, sizeof(tuple_t) * b->size);
            gpuErrChk(cudaMemcpyAsync(data_cpu, b->data_gpu, b->size * sizeof(tuple_t), cudaMemcpyDeviceToHost, cudaStream));
            gpuErrChk(cudaStreamSynchronize(cudaStream));
            for (size_t i=0; i<b->size; i++) {
                tuple_t *t = &(data_cpu[i]);
                cout << "Tuple: " << t->key << " " << t->property_value << " " << t->id << endl;
                if (received + i >= 100)
                    break;
            }
            cudaFreeHost(data_cpu);
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
			if (key_occ.find(get<DEVICE_ID_FIELD>(r)) == key_occ.end())
				key_occ.insert(make_pair(get<DEVICE_ID_FIELD>(r), 0));
		} else {
			incomplete_records++;
		}
		all_records++;
	}
	file.close();
}

void create_tuples(int num_keys) {
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, num_keys - 1);
	mt19937                                                  rng;
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
		t.id = 0;
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
		        new dummy_mi(), new Filter(i, map_degree, app_start_time, batch_size, num_keys), true,
		        true);
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
