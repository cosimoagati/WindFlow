#ifndef COMMON_HPP
#define COMMON_HPP

#include "zipf.hpp"
#include <ff/ff.hpp>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// number of CUDA threads per block
static constexpr auto threads_per_block = 256;

// assert function on GPU
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

// gpuErrChk macro
#define gpuErrChk(ans)                                                                                       \
	{ gpuAssert((ans), __FILE__, __LINE__); }

// struct used by sort_by_key
struct key_less_than_op {
	std::size_t n_dest;

	key_less_than_op(std::size_t _n_dest);

	/* k1 is smaller than k2 if k1 is directed to a destination whose identifier is smaller
	   than the one of k2 */
	__host__ __device__ bool operator()(const std::size_t k1, const std::size_t k2);
};

// struct used by reduce_by_key
struct key_equal_to_op {
	std::size_t n_dest;

	key_equal_to_op(std::size_t _n_dest);

	// k1 and k2 are equal if they are directed to the same destination
	__host__ __device__ bool operator()(const std::size_t k1, const std::size_t k2);
};

// struct of the input tuple
struct tuple_t {
	std::size_t  key;
	std::int64_t value;

	__host__ __device__ tuple_t();

	tuple_t(size_t _key, int64_t _value);
};

// struct of a batch
struct batch_t {
	std::size_t               size;
	tuple_t *                 data_gpu;
	std::size_t *             keys_gpu;
	tuple_t *                 raw_data_gpu;
	std::size_t *             raw_keys_gpu;
	bool                      cleanup = true;
	std::atomic<std::size_t> *counter = nullptr;

	batch_t(std::size_t _size);

	batch_t(std::size_t _size, tuple_t *_data_gpu, std::size_t *_keys_gpu, std::size_t _offset,
	        std::atomic<std::size_t> *_counter);

	~batch_t();
};

// state per key for the stateful processing
struct state_t {
	int index;
	int buffer[100];

	__host__ __device__ state_t();
};

template<typename RandomDistribution = std::uniform_int_distribution<std::mt19937::result_type>>
class Source : public ff::ff_node_t<batch_t> {
public:
	long stream_len;
	long num_keys;
	long batch_len;
	// std::uniform_int_distribution<std::mt19937::result_type> dist;
	// std::geometric_distribution<std::size_t>                 dist;
	// zipf_distribution<std::size_t>                           dist;
	RandomDistribution                                       dist;
	std::uniform_int_distribution<std::mt19937::result_type> dist2;
	std::mt19937                                             rng;
	std::mt19937                                             rng2;
	cudaStream_t                                             cudaStream;
	std::size_t *                                            keys_cpu = nullptr;
	tuple_t *                                                data_cpu = nullptr;

	Source(const long _stream_len, const long _num_keys, const long _batch_len)
	        : stream_len(_stream_len), num_keys(_num_keys), batch_len(_batch_len), dist(0, _num_keys -1),
	          dist2(0, 2000) {
		// set random seed
		// rng.seed(random_device()());
		// rng2.seed(random_device()());
		rng.seed(0);
		rng2.seed(0);
		cudaMallocHost(&keys_cpu, sizeof(size_t) * batch_len);
		cudaMallocHost(&data_cpu, sizeof(tuple_t) * batch_len);
		// initialize CUDA stream
		gpuErrChk(cudaStreamCreate(&cudaStream));
	}

	~Source() {
		cudaFreeHost(keys_cpu);
		cudaFreeHost(data_cpu);
		gpuErrChk(cudaStreamDestroy(cudaStream));
	}

	batch_t *svc(batch_t *) {
		// generation loop
		long sent = 0;
		while (sent < stream_len) {
			const auto b = create_batch();
			this->ff_send_out(b);
			sent += batch_len;
		}
		return this->EOS;
	}

	// function to generate and send a batch of data
	batch_t *create_batch() {
		const auto b = new batch_t(batch_len);
		// cout << "Source Keys [ ";
		for (size_t i = 0; i < batch_len; i++) {
			keys_cpu[i]       = dist(rng);
			data_cpu[i].value = dist2(rng2);
			// cout << keys_cpu[i] << " ";
		}
		// cout << "]" << endl;
		// copy keys_cpu to keys_gpu
		gpuErrChk(cudaMemcpyAsync(b->keys_gpu, keys_cpu, batch_len * sizeof(size_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		gpuErrChk(cudaMemcpyAsync(b->data_gpu, data_cpu, batch_len * sizeof(tuple_t),
		                          cudaMemcpyHostToDevice, cudaStream));
		gpuErrChk(cudaStreamSynchronize(cudaStream));
		return b;
	}
};

class Emitter : public ff::ff_monode_t<batch_t> {
public:
	size_t        n_dest;
	size_t        batch_len;
	cudaStream_t  cudaStream;
	unsigned long tot_elapsed_nsec = 0;
	unsigned long received         = 0;
	size_t *      unique_dests_cpu = nullptr;
	int *         freqs_dests_cpu  = nullptr;

	Emitter(size_t _n_dest, size_t _batch_len);
	~Emitter();
	batch_t *svc(batch_t *b);
	void     svc_end();
};

inline unsigned long current_time_usecs() {
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec) * 1000000L + (t.tv_nsec / 1000);
}

// Sink class
class Sink : public ff::ff_minode_t<batch_t> {
public:
	std::size_t            received        = 0; // counter of received tuples
	std::size_t            received_sample = 0; // counter of recevied tuples per sample
	volatile unsigned long start_time_us;       // starting time usec
	volatile unsigned long last_time_us;        //  time usec
	cudaStream_t           cudaStream;
	unsigned long          correctness = 0;

	Sink();
	~Sink();
	batch_t *svc(batch_t *const b);
	void     svc_end();
};

// function to get the time from the epoch in nanoseconds
inline unsigned long current_time_nsecs() {
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec) * 1000000000L + t.tv_nsec;
}

// processing function on a tuple and the state of its key
// __device__ void process(tuple_t *const t, state_t *const state) {
// 	auto &index           = state->index;
// 	auto &buffer_location = (state->buffer)[index];
// 	auto &value           = t->value;

// 	if (value % 2 == 0) { // even
// 		if (value % 4 == 0) {
// 			buffer_location += value;
// 			value = buffer_location;
// 			index = (index + 1) % 100;
// 		} else {
// 			buffer_location *= value;
// 			value = buffer_location;
// 			index = (index + 2) % 100;
// 		}
// 	} else { // odd
// 		if (value % 5 == 0) {
// 			buffer_location *= value;
// 			value = buffer_location;
// 			index = (index + 3) % 100;
// 		} else {
// 			buffer_location += value;
// 			value = buffer_location;
// 			index = (index + 4) % 100;
// 		}
// 	}
// }

// CUDA Kernel Stateful_Processing_Kernel
__global__ void Stateful_Processing_Kernel(tuple_t *tuples, std::size_t *keys, state_t **states,
                                           std::size_t len);

template<typename Worker,
         typename RandomDistribution = std::uniform_int_distribution<std::mt19937::result_type>>
int run_test(const int argc, char *argv[]) {
	int         option     = 0;
	std::size_t stream_len = 0;
	std::size_t n_keys     = 1;
	std::size_t batch_len  = 1;
	std::size_t par_degree = 1;
	// arguments from command line
	if (argc != 9) {
		std::cout << argv[0] << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]"
		          << std::endl;
		std::exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "l:k:b:n:")) != -1) {
		switch (option) {
		case 'l':
			stream_len = std::atoi(optarg);
			break;
		case 'k':
			n_keys = std::atoi(optarg);
			break;
		case 'b':
			batch_len = std::atoi(optarg);
			break;
		case 'n':
			par_degree = std::atoi(optarg);
			break;
		default: {
			std::cout << argv[0]
			          << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]"
			          << std::endl;
			std::exit(EXIT_SUCCESS);
		}
		}
	}
	auto pipe   = new ff::ff_pipeline();
	auto source = new Source<RandomDistribution>(stream_len, n_keys, batch_len);
	pipe->add_stage(source);
	auto farm = new ff::ff_farm();
	farm->add_emitter(new Emitter(par_degree, batch_len));

	std::vector<ff::ff_node *> ws;
	for (std::size_t i = 0; i < par_degree; i++)
		ws.push_back(new Worker(par_degree, i));
	farm->add_workers(ws);
	farm->add_collector(new Sink());

	pipe->add_stage(farm);
	std::cout << "Start..." << std::endl;
	pipe->run_and_wait_end();
	std::cout << "...end" << std::endl;
	return 0;
}

#endif
