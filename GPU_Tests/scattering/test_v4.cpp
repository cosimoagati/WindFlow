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
 *  Test of the scatter functionality (version 4). This implementation is based
 *  on an in-place sort_by_key of the input batch, in order to place all the
 *  inputs directed to the same destination in contiguous positions. Then, each
 *  destination receives a batch pointing to the same GPU area (but they are able
 *  to access only the right inputs in the batch). This avoids paying the copy
 *  overhead of creating sub-batches.
 */ 

// includes
#include<random>
#include<iostream>
#include<ff/ff.hpp>
#include<sys/time.h>
#include<sys/stat.h>
#include<thrust/sort.h>
#include<thrust/count.h>
#include<thrust/reduce.h>
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>

using namespace std;
using namespace ff;

// function to get the time from the epoch in microseconds
inline unsigned long current_time_usecs() __attribute__((always_inline));
inline unsigned long current_time_usecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000L + (t.tv_nsec / 1000);
}

// function to get the time from the epoch in nanoseconds
inline unsigned long current_time_nsecs() __attribute__((always_inline));
inline unsigned long current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

// assert function on GPU
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=false)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

// gpuErrChk macro
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// struct used by sort_by_key
struct key_less_than_op
{
    size_t n_dest;

    // constructor
    key_less_than_op(size_t _n_dest):
              n_dest(_n_dest) {}

    /* k1 is smaller than k2 if k1 is directed to a destination whose identifier is smaller
       than the one of k2 */
    __host__ __device__ bool operator()(const size_t k1, const size_t k2)
    {
        return ((k1 % n_dest) < (k2 % n_dest));
    }
};

// struct used by reduce_by_key
struct key_equal_to_op
{
    size_t n_dest;

    // constructor
    key_equal_to_op(size_t _n_dest):
              n_dest(_n_dest) {}

    // k1 and k2 are equal if they are directed to the same destination
    __host__ __device__ bool operator()(const size_t k1, const size_t k2)
    {
        return ((k1 % n_dest) == (k2 % n_dest));
    }
};

// struct of the input tuple
struct tuple_t
{
    size_t key;
    int64_t value;

    // constructor I
    __host__ __device__ tuple_t(): key(0), value(0) {}

    // constructor II
    tuple_t(size_t _key,
            int64_t _value):
            key(_key),
            value(_value) {}
};

// struct of a batch
struct batch_t
{
    size_t size;
    tuple_t *data_gpu;
    size_t *keys_gpu;
    tuple_t *raw_data_gpu;
    size_t *raw_keys_gpu;
    bool cleanup=true;
    atomic<size_t> *counter=nullptr;

    // constructor I
    batch_t(size_t _size):
            size(_size)
    {
        cudaMalloc(&data_gpu, size * sizeof(tuple_t));
        cudaMalloc(&keys_gpu, size * sizeof(size_t));
        raw_data_gpu = data_gpu;
        raw_keys_gpu = keys_gpu;
    }

    // constructor II
    batch_t(size_t _size,
            tuple_t *_data_gpu,
            size_t *_keys_gpu,
            size_t _offset,
            atomic<size_t> *_counter)
    {
        size = _size;
        raw_data_gpu = _data_gpu;
        raw_keys_gpu = _keys_gpu;
        data_gpu = raw_data_gpu + _offset;
        keys_gpu = raw_keys_gpu + _offset;
        counter = _counter;
    }

    // destructor
    ~batch_t()
    {
        if (counter != nullptr) {
            size_t old_cnt = counter->fetch_sub(1);
            if (old_cnt == 1) {
                if (cleanup) {
                    cudaFree(raw_data_gpu);
                    cudaFree(raw_keys_gpu);
                }
                delete counter;
            }
        }
        else if (cleanup) {
            cudaFree(raw_data_gpu);
            cudaFree(raw_keys_gpu);
        }
    }
};

// Source class
class Source: public ff_node_t<batch_t>
{
public:
    long stream_len;
    long num_keys;
    long batch_len;
    std::uniform_int_distribution<std::mt19937::result_type> dist;
    mt19937 rng;
    cudaStream_t cudaStream;
    size_t *keys_cpu=nullptr;

    // constructor
    Source(long _stream_len,
           long _num_keys,
           long _batch_len):
           stream_len(_stream_len),
           num_keys(_num_keys),
           batch_len(_batch_len),
           dist(0, _num_keys-1)
    {
        // set random seed
        //rng.seed(std::random_device()());
        rng.seed(0);
        // allocate keys_cpu array
        cudaMallocHost(&keys_cpu, sizeof(size_t) * batch_len);
        // initialize CUDA stream
        gpuErrChk(cudaStreamCreate(&cudaStream));
    }

    // destructor
    ~Source()
    {
        cudaFreeHost(keys_cpu);
        gpuErrChk(cudaStreamDestroy(cudaStream));
    }

    // svc method
    batch_t *svc(batch_t *)
    {
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
    batch_t *create_batch()
    {
        batch_t *b = new batch_t(batch_len);
        //std::cout << "Source Keys [ ";
        for (size_t i=0; i<batch_len; i++) {
            keys_cpu[i] = dist(rng);
            //std::cout << keys_cpu[i] << " ";
        }
        //std::cout << "]" << std::endl;
        // copy keys_cpu to keys_gpu
        gpuErrChk(cudaMemcpyAsync(b->keys_gpu, keys_cpu, batch_len * sizeof(size_t), cudaMemcpyHostToDevice, cudaStream));
        // data_gpu is not initialized because it is not used by this benchmark
        gpuErrChk(cudaStreamSynchronize(cudaStream));
        return b;
    }
};

// Emitter class
class Emitter: public ff_monode_t<batch_t>
{
public:
    size_t n_dest;
    size_t batch_len;
    cudaStream_t cudaStream;
    unsigned long tot_elapsed_nsec=0;
    unsigned long received=0;
    size_t *unique_dests_cpu=nullptr;
    int *freqs_dests_cpu=nullptr;

    // constructor
    Emitter(size_t _n_dest,
            size_t _batch_len):
            n_dest(_n_dest),
            batch_len(_batch_len)
    {
        // allocate unique_keys_cpu array
        cudaMallocHost(&unique_dests_cpu, sizeof(size_t) * n_dest);
        // allocate freqs_keys_cpu array
        cudaMallocHost(&freqs_dests_cpu, sizeof(int) * n_dest);
        // initialize CUDA stream
        gpuErrChk(cudaStreamCreate(&cudaStream));
    }

    // destructor
    ~Emitter()
    {
        cudaFreeHost(unique_dests_cpu);
        cudaFreeHost(freqs_dests_cpu);
        gpuErrChk(cudaStreamDestroy(cudaStream));
    }

    // svc method
    batch_t *svc(batch_t *b)
    {
        volatile unsigned long start_time_nsec = current_time_nsecs();
        received++;
        if (n_dest == 1) {
            volatile unsigned long end_time_nsec = current_time_nsecs();
            unsigned long elapsed_time_nsec = end_time_nsec - start_time_nsec;
            tot_elapsed_nsec += elapsed_time_nsec;
            return b;
        }
        // sort of the input batch: inputs directed to the same destination are placed contiguously in the batch
        thrust::device_ptr<tuple_t> th_data_gpu = thrust::device_pointer_cast(b->data_gpu);
        thrust::device_ptr<size_t> th_keys_gpu = thrust::device_pointer_cast(b->keys_gpu);
        thrust::sort_by_key(thrust::cuda::par.on(cudaStream), th_keys_gpu, th_keys_gpu + b->size, th_data_gpu, key_less_than_op(n_dest));
        // compute the number of unique occurrences of each destination identifier
        thrust::device_vector<int> ones_gpu(b->size); // for sure they are not more than n_dest
        thrust::fill(ones_gpu.begin(), ones_gpu.end(), 1);
        thrust::device_vector<size_t> unique_dests_gpu(n_dest); // for sure they are not more than n_dest
        thrust::device_vector<int> freqs_dests_gpu(n_dest); // for sure they are not more than n_dest
        auto end = thrust::reduce_by_key(thrust::cuda::par.on(cudaStream), th_keys_gpu, th_keys_gpu + b->size, ones_gpu.begin(), unique_dests_gpu.begin(), freqs_dests_gpu.begin(), key_equal_to_op(n_dest));
        size_t num_found_dests = end.first - unique_dests_gpu.begin();
        assert(num_found_dests > 0);
        gpuErrChk(cudaMemcpyAsync(unique_dests_cpu, thrust::raw_pointer_cast(unique_dests_gpu.data()), sizeof(size_t) * num_found_dests, cudaMemcpyDeviceToHost, cudaStream));
        gpuErrChk(cudaMemcpyAsync(freqs_dests_cpu, thrust::raw_pointer_cast(freqs_dests_gpu.data()), sizeof(int) * num_found_dests, cudaMemcpyDeviceToHost, cudaStream));
        gpuErrChk(cudaStreamSynchronize(cudaStream));
        size_t offset = 0;
        atomic<size_t> *counter = new atomic<size_t>(num_found_dests); // to deallocate correctly the GPU buffer within the batch
        // for each destination that must receive data
        for (size_t i=0; i<num_found_dests; i++) {
            int result = freqs_dests_cpu[i];
            batch_t *bout = new batch_t(result, b->data_gpu, b->keys_gpu, offset, counter);
            this->ff_send_out_to(bout, unique_dests_cpu[i] % n_dest);
            offset += result;
        }
        b->cleanup = false; // we don't want to deallocate the input GPU buffer because it is used by the transmitted batches
        delete b;
        volatile unsigned long end_time_nsec = current_time_nsecs();
        unsigned long elapsed_time_nsec = end_time_nsec - start_time_nsec;
        tot_elapsed_nsec += elapsed_time_nsec;
        return this->GO_ON;
    }

    // svc_end method
    void svc_end()
    {
        std::cout << "[Emitter] average service time: " << (((double) tot_elapsed_nsec)/received) / 1000 << " usec" << std::endl;
    }
};

// Worker class
class Worker: public ff_node_t<batch_t>
{
public:
    size_t received; // counter of received tuples
    size_t par_deg;
    size_t id;
    cudaStream_t cudaStream;
    size_t *keys_cpu=nullptr;

    // constructor
    Worker(size_t _par_deg,
           size_t _id):
           received(0),
           par_deg(_par_deg),
           id(_id)
    {
        // initialize CUDA stream
        gpuErrChk(cudaStreamCreate(&cudaStream));       
    }

    // destructor
    ~Worker()
    {
        if (keys_cpu != nullptr) {
            cudaFreeHost(keys_cpu);
        }
        gpuErrChk(cudaStreamDestroy(cudaStream));
    }

    // svc method
    batch_t *svc(batch_t *b)
    {
#if 1
        received += b->size;
        cudaMallocHost(&keys_cpu, sizeof(size_t) * b->size);
        // copy keys_gpu to keys_cpu
        gpuErrChk(cudaMemcpyAsync(keys_cpu, b->keys_gpu, b->size * sizeof(size_t), cudaMemcpyDeviceToHost, cudaStream));
        gpuErrChk(cudaStreamSynchronize(cudaStream));
        // check the correctness of the received batch
        //std::cout << "Worker " << id << ", Keys [ ";
        for (size_t i=0; i<b->size; i++) {
            assert(keys_cpu[i] % par_deg == id);
            //std::cout << keys_cpu[i] << " ";
        }
        //std::cout << "]" << std::endl;
        cudaFreeHost(keys_cpu);
#endif
        return b;
    }
};

// Sink class
class Sink: public ff_minode_t<batch_t>
{
public:
    size_t received; // counter of received tuples
    size_t received_sample; // counter of recevied tuples per sample
    volatile unsigned long start_time_us; // starting time usec
    volatile unsigned long last_time_us; //  time usec

    // constructor
    Sink(): received(0),
            received_sample(0),
            start_time_us(current_time_usecs()),
            last_time_us(current_time_usecs()) {}

    // svc method
    batch_t *svc(batch_t *b)
    {
        received += b->size;
        received_sample += b->size;
        // print the throughput per second
        if (current_time_usecs() - last_time_us > 1000000) {
            double elapsed_sec = ((double) (current_time_usecs() - start_time_us)) / 1000000;
            std::cout << "[SINK] time " << elapsed_sec << " received " << received_sample << std::endl;
            received_sample = 0;
            last_time_us = current_time_usecs();
        }
        delete b;
        return this->GO_ON;
    }

    void svc_end()
    {
        double elapsed_sec = ((double) (current_time_usecs() - start_time_us)) / 1000000;
        std::cout << "[SINK] time " << elapsed_sec << " received " << received_sample << std::endl;
        std::cout << "[SINK] total received " << received << std::endl;
    }
};

// main
int main(int argc, char * argv[]) {
    int option = 0;
    size_t stream_len = 0;
    size_t n_keys = 1;
    size_t batch_len = 1;
    size_t par_degree = 1;
    // arguments from command line
    if (argc != 9) {
        cout << argv[0] << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]" << endl;
        exit(EXIT_SUCCESS);
    }
    while ((option = getopt(argc, argv, "l:k:b:n:")) != -1) {
        switch (option) {
            case 'l': stream_len = atoi(optarg);
                     break;
            case 'k': n_keys = atoi(optarg);
                     break;
            case 'b': batch_len = atoi(optarg);
                     break;
            case 'n': par_degree = atoi(optarg);
                     break;
            default: {
                cout << argv[0] << " -l [stream_length] -k [n_keys] -b [batch length] -n [par degree]" << endl;
                exit(EXIT_SUCCESS);
            }
        }
    }
    ff_pipeline *pipe = new ff_pipeline();
    Source *source = new Source(stream_len, n_keys, batch_len);
    pipe->add_stage(source);
    ff_farm *farm = new ff_farm();
    farm->add_emitter(new Emitter(par_degree, batch_len));
    std::vector<ff_node *> ws;
    for (size_t i=0; i<par_degree; i++) {
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
