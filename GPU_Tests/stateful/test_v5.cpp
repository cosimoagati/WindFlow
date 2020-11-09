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
 *  Test of the stateful batch computation (version 5). This implementation is
 *  identical to version 4 but exploit overlapping of kernel processing with
 *  the allocation of state objects for the next batch.
 */

// includes
#include "zipf.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <unordered_map>

using namespace std;
using namespace ff;

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

// assert function on GPU
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			exit(code);
		}
	}
}

// gpuErrChk macro
#define gpuErrChk(ans)                                                                                       \
	{ gpuAssert((ans), __FILE__, __LINE__); }

// functor to compare two keys (less than)
struct key_less_than_op {
	size_t n_dest;

	// constructor
	key_less_than_op(size_t _n_dest) : n_dest(_n_dest) {}

	__host__ __device__ bool operator()(const size_t k1, const size_t k2) {
		return ((k1 % n_dest) < (k2 % n_dest));
	}
};

// functor to compare two keys (equality)
struct key_equal_to_op {
	size_t n_dest;

	// constructor
	key_equal_to_op(size_t _n_dest) : n_dest(_n_dest) {}

	__host__ __device__ bool operator()(const size_t k1, const size_t k2) {
		return ((k1 % n_dest) == (k2 % n_dest));
	}
};

// struct of the input tuple
struct tuple_t {
	size_t  key;
	int64_t value;

	// constructor I
	__host__ __device__ tuple_t() : key(0), value(0) {}

	// constructor II
	tuple_t(size_t _key, int64_t _value) : key(_key), value(_value) {}
};

// struct of a batch
struct batch_t {
	size_t          size;
	tuple_t *       data_gpu;
	size_t *        keys_gpu;
	size_t *        keys_cpu;
	tuple_t *       raw_data_gpu;
	size_t *        raw_keys_gpu;
	size_t *        raw_keys_cpu;
	bool            cleanup = true;
	atomic<size_t> *counter = nullptr;

	// constructor I
	batch_t(size_t _size) : size(_size) {
		cudaMalloc(&data_gpu, size * sizeof(tuple_t));
		cudaMalloc(&keys_gpu, size * sizeof(size_t));
		cudaMallocHost(&keys_cpu, sizeof(size_t) * size);
		raw_data_gpu = data_gpu;
		raw_keys_gpu = keys_gpu;
		raw_keys_cpu = keys_cpu;
	}

	// constructor II
	batch_t(size_t _size, tuple_t *_data_gpu, size_t *_keys_gpu, size_t *_keys_cpu, size_t _offset,
	        atomic<size_t> *_counter) {
		size         = _size;
		raw_data_gpu = _data_gpu;
		raw_keys_gpu = _keys_gpu;
		raw_keys_cpu = _keys_cpu;
		data_gpu     = raw_data_gpu + _offset;
		keys_gpu     = raw_keys_gpu + _offset;
		keys_cpu     = raw_keys_cpu + _offset;
		counter      = _counter;
	}

	// destructor
	~batch_t() {
		if (counter != nullptr) {
			size_t old_cnt = counter->fetch_sub(1);
			if (old_cnt == 1) {
				if (cleanup) {
					cudaFree(raw_data_gpu);
					cudaFree(raw_keys_gpu);
					cudaFreeHost(raw_keys_cpu);
				}
				delete counter;
			}
		} else if (cleanup) {
			cudaFree(raw_data_gpu);
			cudaFree(raw_keys_gpu);
			cudaFreeHost(raw_keys_cpu);
		}
	}
};

// state per key for the stateful processing
struct state_t {
	int index;
	int buffer[100];

	// constructor
	__host__ __device__ state_t() {
		index = 0;
		memset(buffer, 0, sizeof(int) * 100);
	}
};

__device__ void process(tuple_t *t, state_t *state) {
	auto &index           = state->index;
	auto &buffer_location = (state->buffer)[index];
	auto &value           = t->value;

	if (value % 2 == 0) { // even
		if (value % 4 == 0) {
			buffer_location += value;
			value = buffer_location;
			index = (index + 1) % 100;
		} else {
			buffer_location *= value;
			value = buffer_location;
			index = (index + 2) % 100;
		}
	} else { // odd
		if (value % 5 == 0) {
			buffer_location *= value;
			value = buffer_location;
			index = (index + 3) % 100;
		} else {
			buffer_location += value;
			value = buffer_location;
			index = (index + 4) % 100;
		}
	}
}

// CUDA Kernel Stateful_Processing_Kernel
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, size_t *keys, size_t *dist_keys_gpu,
                                           size_t num_dist_keys, state_t **states, size_t len,
                                           int num_active_thread_per_warp) {
	const int id = threadIdx.x + blockIdx.x * blockDim.x; // id of the thread in the kernel
	const int threads_per_worker =
	        warpSize / num_active_thread_per_warp;  // number of threads composing a worker entity
	const int num_threads = gridDim.x * blockDim.x; // number of threads in the kernel
	const int num_workers = num_threads / threads_per_worker; // number of workers
	const int id_worker   = id / threads_per_worker;          // id of the worker
	// only the first thread of each warp works, the others are idle
	if (id % threads_per_worker == 0) {
		for (size_t id_key = id_worker; id_key < num_dist_keys; id_key += num_workers) {
			const auto key = dist_keys_gpu[id_key]; // key used
			for (size_t i = 0; i < len; i++) {
				if (key == keys[i]) {
					process(&(tuples[i]), states[i]);
				}
			}
		}
	}
}

// Source class
class Source : public ff_node_t<batch_t> {
public:
	long stream_len;
	long num_keys;
	long batch_len;
	// std::uniform_int_distribution<std::mt19937::result_type> dist;
	shifted_zipf_distribution<std::size_t>                   dist;
	std::uniform_int_distribution<std::mt19937::result_type> dist2;
	mt19937                                                  rng;
	mt19937                                                  rng2;
	cudaStream_t                                             cudaStream;
	size_t *                                                 keys_cpu = nullptr;
	tuple_t *                                                data_cpu = nullptr;

	// constructor
	Source(long _stream_len, long _num_keys, long _batch_len)
	        : stream_len(_stream_len), num_keys(_num_keys), batch_len(_batch_len), dist(0, _num_keys - 1),
	          dist2(0, 2000) {
		// set random seed
		// rng.seed(std::random_device()());
		// rng2.seed(std::random_device()());
		rng.seed(0);
		rng2.seed(0);
		cudaMallocHost(&keys_cpu, sizeof(size_t) * batch_len);
		cudaMallocHost(&data_cpu, sizeof(tuple_t) * batch_len);
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	// destructor
	~Source() {
		cudaFreeHost(keys_cpu);
		cudaFreeHost(data_cpu);
		gpuErrChk(cudaStreamDestroy(cudaStream));
	}

	// svc method
	batch_t *svc(batch_t *) {
		// generation loop
		long sent = 0;
		while (sent < stream_len) {
			batch_t *b = create_batch();
			this->ff_send_out(b);
			sent += batch_len;
		}
		return this->EOS;
	}

	// function to generate and send a batch of data
	batch_t *create_batch() {
		batch_t *b = new batch_t(batch_len);
		// std::cout << "Source Keys [ ";
		for (size_t i = 0; i < batch_len; i++) {
			keys_cpu[i]       = dist(rng);
			data_cpu[i].value = dist2(rng2);
			// std::cout << keys_cpu[i] << " ";
		}
		// std::cout << "]" << std::endl;
		// copy keys_cpu to keys_gpu
		gpuErrChk(cudaMemcpyAsync(b->data_gpu, data_cpu, batch_len * sizeof(tuple_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		gpuErrChk(cudaMemcpyAsync(b->keys_gpu, keys_cpu, batch_len * sizeof(size_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		memcpy(b->keys_cpu, keys_cpu, sizeof(size_t) * batch_len);
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		return b;
	}
};

// Emitter class
class Emitter : public ff_monode_t<batch_t> {
public:
	size_t        n_dest;
	size_t        batch_len;
	cudaStream_t  cudaStream;
	unsigned long tot_elapsed_nsec = 0;
	unsigned long received         = 0;
	size_t *      unique_dests_cpu = nullptr;
	int *         freqs_dests_cpu  = nullptr;

	// constructor
	Emitter(size_t _n_dest, size_t _batch_len) : n_dest(_n_dest), batch_len(_batch_len) {
		// allocate unique_keys_cpu array
		cudaMallocHost(&unique_dests_cpu, sizeof(size_t) * n_dest);
		// allocate freqs_keys_cpu array
		cudaMallocHost(&freqs_dests_cpu, sizeof(int) * n_dest);
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	// destructor
	~Emitter() {
		cudaFreeHost(unique_dests_cpu);
		cudaFreeHost(freqs_dests_cpu);
		gpuErrChk(cudaStreamDestroy(cudaStream));
	}

	// svc method
	batch_t *svc(batch_t *const b) {
		volatile unsigned long start_time_nsec = current_time_nsecs();
		received++;
		if (n_dest == 1) {
			volatile unsigned long end_time_nsec     = current_time_nsecs();
			unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
			tot_elapsed_nsec += elapsed_time_nsec;
			return b;
		}
		// sort of the input batch: inputs directed to the same destination are placed contiguously in
		// the batch
		const auto th_data_gpu = thrust::device_pointer_cast(b->data_gpu);
		const auto th_keys_gpu = thrust::device_pointer_cast(b->keys_gpu);
		thrust::sort_by_key(thrust::cuda::par.on(cudaStream), th_keys_gpu, th_keys_gpu + b->size,
		                    th_data_gpu, key_less_than_op(n_dest));
		gpuErrChk(cudaMemcpyAsync(b->keys_cpu, b->keys_gpu, sizeof(size_t) * b->size,
		                          cudaMemcpyDeviceToHost, cudaStream));
		// compute the number of unique occurrences of each destination identifier
		thrust::device_vector<int> ones_gpu(b->size); // for sure they are not more than n_dest
		thrust::fill(ones_gpu.begin(), ones_gpu.end(), 1);
		thrust::device_vector<size_t> unique_dests_gpu(
		        n_dest);                                    // for sure they are not more than n_dest
		thrust::device_vector<int> freqs_dests_gpu(n_dest); // for sure they are not more than n_dest
		const auto   end = thrust::reduce_by_key(thrust::cuda::par.on(cudaStream), th_keys_gpu,
                                                       th_keys_gpu + b->size, ones_gpu.begin(),
                                                       unique_dests_gpu.begin(), freqs_dests_gpu.begin(),
                                                       key_equal_to_op(n_dest));
		const size_t num_found_dests = end.first - unique_dests_gpu.begin();
		assert(num_found_dests > 0);
		gpuErrChk(cudaMemcpyAsync(unique_dests_cpu, thrust::raw_pointer_cast(unique_dests_gpu.data()),
		                          sizeof(size_t) * num_found_dests, cudaMemcpyDeviceToHost,
		                          cudaStream));
		gpuErrChk(cudaMemcpyAsync(freqs_dests_cpu, thrust::raw_pointer_cast(freqs_dests_gpu.data()),
		                          sizeof(int) * num_found_dests, cudaMemcpyDeviceToHost, cudaStream));
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		size_t offset  = 0;
		auto   counter = new atomic<size_t>(
                        num_found_dests); // to deallocate correctly the GPU buffer within the batch
		// for each destination that must receive data
		for (size_t i = 0; i < num_found_dests; i++) {
			int        result = freqs_dests_cpu[i];
			const auto bout =
			        new batch_t(result, b->data_gpu, b->keys_gpu, b->keys_cpu, offset, counter);
			this->ff_send_out_to(bout, unique_dests_cpu[i] % n_dest);
			offset += result;
		}
		b->cleanup = false; // we don't want to deallocate the input GPU buffer because it is used by
		                    // the transmitted batches
		delete b;
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		return this->GO_ON;
	}

	// svc_end method
	void svc_end() {
		std::cout << "[Emitter] average service time: "
		          << (((double) tot_elapsed_nsec) / received) / 1000 << " usec" << std::endl;
	}
};

// Worker class
class Worker : public ff_node_t<batch_t> {
public:
	unsigned long                    received_batch = 0;
	size_t                           received       = 0; // counter of received tuples
	size_t                           par_deg;
	size_t                           id;
	cudaStream_t                     cudaStream;
	unordered_map<size_t, state_t *> hashmap;
	unsigned long                    tot_elapsed_nsec   = 0;
	int                              numSMs             = 0;
	int                              max_threads_per_sm = 0;
	int                              max_blocks_per_sm  = 0;
	int                              threads_per_warp   = 0;
	batch_t *                        batch_to_be_sent   = nullptr;
	state_t **                       state_ptrs_gpu     = nullptr;
	size_t *                         dist_keys_gpu      = nullptr;

	// constructor
	Worker(size_t _par_deg, size_t _id) : par_deg(_par_deg), id(_id) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
		// get some configuration parameters of the GPU
		gpuErrChk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
		gpuErrChk(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
		                                 0));
		gpuErrChk(
		        cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
		gpuErrChk(cudaDeviceGetAttribute(&threads_per_warp, cudaDevAttrWarpSize, 0));
		assert(numSMs > 0);             // 15
		assert(max_threads_per_sm > 0); // 2048
		assert(max_blocks_per_sm > 0);  // 16
		assert(threads_per_warp > 0);   // 32
		if (id == 0) {
			cout << "NumSMs: " << numSMs << ", threads per SM: " << max_threads_per_sm
			     << ", blocks per SM: " << max_blocks_per_sm << ", threads per warp "
			     << threads_per_warp << endl;
		}
	}

	// destructor
	~Worker() { gpuErrChk(cudaStreamDestroy(cudaStream)); }

	// svc method
	batch_t *svc(batch_t *b) {
		volatile unsigned long start_time_nsec = current_time_nsecs();
		received_batch++;
		received += b->size;
#if 1
		// check the correctness of the received batch
		// std::cout << "Worker " << id << ", Keys [ ";
		for (size_t i = 0; i < b->size; i++) {
			assert(b->keys_cpu[i] % par_deg == id);
			// std::cout << b->keys_cpu[i] << " ";
		}
		// std::cout << "]" << std::endl;
#endif
		// adds and lookups in the hashmap
		const auto state_ptrs_cpu = (state_t **) malloc(b->size * sizeof(state_t *));
		int        num_dist_keys  = 0;
		unordered_map<size_t, size_t> dist_map;
		const auto                    dist_keys_cpu = (size_t *) malloc(b->size * sizeof(size_t));
		for (size_t i = 0; i < b->size; i++) {
			const auto key = b->keys_cpu[i];
			auto       it  = hashmap.find(key);
			if (it == hashmap.end()) {
				// create the state of that key
				state_t *state = nullptr;
				cudaMalloc(&state, sizeof(state_t));
				cudaMemsetAsync(state, 0, sizeof(state_t), cudaStream);
				hashmap.insert(std::make_pair(key, state));
				it = hashmap.find(key);
			}
			state_ptrs_cpu[i] = (*it).second;
			// count distinct keys in the batch
			const auto it2 = dist_map.find(key);
			if (it2 == dist_map.end()) {
				dist_map.insert(std::make_pair(key, 1));
				dist_keys_cpu[num_dist_keys] = key;
				num_dist_keys++;
			}
		}
		if (batch_to_be_sent != nullptr) {
			gpuErrChk(cudaStreamSynchronize(cudaStream));
			cudaFree(state_ptrs_gpu);
			cudaFree(dist_keys_gpu);
			this->ff_send_out(batch_to_be_sent);
		}
		// copy state_ptrs_cpu to state_ptrs_gpu
		cudaMalloc(&state_ptrs_gpu, b->size * sizeof(state_t *));
		gpuErrChk(cudaMemcpyAsync(state_ptrs_gpu, state_ptrs_cpu, b->size * sizeof(state_t *),
		                          cudaMemcpyHostToDevice, cudaStream));
		// copy distinct keys on GPU
		cudaMalloc(&dist_keys_gpu, sizeof(size_t) * num_dist_keys);
		gpuErrChk(cudaMemcpyAsync(dist_keys_gpu, dist_keys_cpu, num_dist_keys * sizeof(size_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		// launch the kernel to compute the results
		const auto warps_per_block = ((max_threads_per_sm / max_blocks_per_sm) / threads_per_warp);
		const auto tot_num_warps   = warps_per_block * max_blocks_per_sm * numSMs;
		// compute how many threads should be active per warps
		int32_t x = (int32_t) ceil(((double) num_dist_keys) / tot_num_warps);
		if (x > 1) {
			x = next_power_of_two(x);
		}
		const auto num_active_thread_per_warp = std::min(x, threads_per_warp);
		const auto num_blocks = std::min((int) ceil(((double) num_dist_keys) / warps_per_block),
		                                 numSMs * max_blocks_per_sm);
		Stateful_Processing_Kernel<<<num_blocks, warps_per_block * threads_per_warp, 0, cudaStream>>>(
		        b->data_gpu, b->keys_gpu, dist_keys_gpu, num_dist_keys, state_ptrs_gpu, b->size,
		        num_active_thread_per_warp);
		free(state_ptrs_cpu);
		free(dist_keys_cpu);
		volatile unsigned long end_time_nsec     = current_time_nsecs();
		unsigned long          elapsed_time_nsec = end_time_nsec - start_time_nsec;
		tot_elapsed_nsec += elapsed_time_nsec;
		batch_to_be_sent = b;
		return this->GO_ON;
	}

	// eosnotify method
	void eosnotify(ssize_t id) {
		if (batch_to_be_sent != nullptr) {
			gpuErrChk(cudaStreamSynchronize(cudaStream));
			cudaFree(state_ptrs_gpu);
			cudaFree(dist_keys_gpu);
			this->ff_send_out(batch_to_be_sent);
		}
	}

	// svc_end method
	void svc_end() {
		printf("[Worker] average service time: %f usec\n",
		       (((double) tot_elapsed_nsec) / received_batch) / 1000);
		// std::cout << "[Worker] average service time: " << (((double)
		// tot_elapsed_nsec)/received_batch) / 1000 << " usec" << std::endl;
	}
};

// Sink class
class Sink : public ff_minode_t<batch_t> {
public:
	size_t                 received        = 0; // counter of received tuples
	size_t                 received_sample = 0; // counter of recevied tuples per sample
	volatile unsigned long start_time_us;       // starting time usec
	volatile unsigned long last_time_us;        //  time usec
	cudaStream_t           cudaStream;
	unsigned long          correctness = 0;

	// constructor
	Sink() : start_time_us(current_time_usecs()), last_time_us(current_time_usecs()) {
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	// destructor
	~Sink() { gpuErrChk(cudaStreamDestroy(cudaStream)); }

	// svc method
	batch_t *svc(batch_t *b) {
		received += b->size;
		received_sample += b->size;
		tuple_t *data_cpu = nullptr;
		cudaMallocHost(&data_cpu, sizeof(tuple_t) * b->size);
		gpuErrChk(cudaMemcpyAsync(data_cpu, b->data_gpu, b->size * sizeof(tuple_t),
		                          cudaMemcpyDeviceToHost, cudaStream));
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		for (size_t i = 0; i < b->size; i++) {
			correctness += data_cpu[i].value;
		}
		// print the throughput per second
		if (current_time_usecs() - last_time_us > 1000000) {
			const double elapsed_sec =
			        ((double) (current_time_usecs() - start_time_us)) / 1000000;
			std::cout << "[SINK] time " << elapsed_sec << " received " << received_sample
			          << std::endl;
			received_sample = 0;
			last_time_us    = current_time_usecs();
		}
		cudaFreeHost(data_cpu);
		delete b;
		return this->GO_ON;
	}

	void svc_end() {
		const double elapsed_sec = ((double) (current_time_usecs() - start_time_us)) / 1000000;
		// std::cout << "[SINK] time " << elapsed_sec << " received " << received_sample << std::endl;
		// std::cout << "[SINK] total received " << received << std::endl;
		printf("[SINK] time %f received %d\n", elapsed_sec, received_sample);
		printf("[SINK] total received %d with total count %ld\n", received, correctness);
	}
};

// main
int main(int argc, char *argv[]) {
	int    option     = 0;
	size_t stream_len = 0;
	size_t n_keys     = 1;
	size_t batch_len  = 1;
	size_t par_degree = 1;
	// arguments from command line
	if (argc != 9) {
		cout << argv[0] << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]"
		     << endl;
		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "l:k:b:n:")) != -1) {
		switch (option) {
		case 'l':
			stream_len = atoi(optarg);
			break;
		case 'k':
			n_keys = atoi(optarg);
			break;
		case 'b':
			batch_len = atoi(optarg);
			break;
		case 'n':
			par_degree = atoi(optarg);
			break;
		default: {
			cout << argv[0] << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]"
			     << endl;
			exit(EXIT_SUCCESS);
		}
		}
	}
	auto pipe   = new ff_pipeline();
	auto source = new Source(stream_len, n_keys, batch_len);
	pipe->add_stage(source);
	auto farm = new ff_farm();
	farm->add_emitter(new Emitter(par_degree, batch_len));
	std::vector<ff_node *> ws;
	for (size_t i = 0; i < par_degree; i++) {
		ws.push_back(new Worker(par_degree, i));
	}
	farm->add_workers(ws);
	farm->add_collector(new Sink());
	pipe->add_stage(farm);
	cout << "Start..." << endl;
	pipe->run_and_wait_end();
	cout << "...end" << endl;
	return 0;
}
