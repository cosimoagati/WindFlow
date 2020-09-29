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
 *  Test of the scatter functionality (version 2). For each input batch, the
 *  emitter node computes for each destination the scan to identify the number
 *  of inputs in the batch to be transmitted to the same destination. Then, all
 *  those inputs are copied in a specific batch to be transmitted to that
 *  destination. This idea (scan and sub-batch creation) is applied to each
 *  destination connected to the emitter. The scan is computed sequentially on
 *  the CPU, while the sub-batch creation is done in parallel on the GPU.
 */ 

// includes
#include<vector>
#include<random>
#include<cstdlib>
#include<iostream>
#include<iostream>
#include<ff/ff.hpp>
#include<sys/time.h>
#include<sys/stat.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

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

// number of CUDA threads per block
static constexpr auto threads_per_block = 256;

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

// struct of the input tuple
struct tuple_t
{
    size_t key;
    int64_t value;

    // constructor
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

    // constructor
    batch_t(size_t _size):
            size(_size)
    {
        cudaMalloc(&data_gpu, size * sizeof(tuple_t));
        cudaMalloc(&keys_gpu, size * sizeof(size_t));
    }

    // destructor
    ~batch_t()
    {
        cudaFree(data_gpu);
        cudaFree(keys_gpu);
    }

    // method to print the keys in the batch
    void print(cudaStream_t &cudaStream)
    {
        size_t *keys_cpu;
        cudaMallocHost(&keys_cpu, sizeof(size_t) * size);
        // copy keys_gpu to keys_cpu
        gpuErrChk(cudaMemcpyAsync(keys_cpu, keys_gpu, size * sizeof(size_t), cudaMemcpyDeviceToHost, cudaStream));
        gpuErrChk(cudaStreamSynchronize(cudaStream));
        cout << "Batch Keys: [ ";
        for (size_t i=0; i<size; i++) {
            cout << keys_cpu[i] << " ";
        }
        cout << "]" << std::endl;
        cudaFreeHost(keys_cpu);
    }
};

// map_to_target kernel
__global__ void map_to_target(int *output,
                              size_t *input,
                              int size,
                              int n_dest,
                              int target_value)
{
    const auto absolute_id  =  blockDim.x * blockIdx.x + threadIdx.x;
    if (absolute_id < size) {
        output[absolute_id] = ((input[absolute_id] % n_dest) == target_value);
    }
}

// mapped_scan function
int mapped_scan(int *output, size_t *input, int size, int n_dest, int target_value, cudaStream_t *cudaStream)
{
    int *output_cpu = nullptr;
    cudaMallocHost(&output_cpu, sizeof(int) * size);
    auto num_of_blocks = ceil(((double) size) / threads_per_block);
    map_to_target<<<num_of_blocks, threads_per_block, 0, *cudaStream>>>(output, input, size, n_dest, target_value);
    assert(cudaGetLastError() == cudaSuccess);
    gpuErrChk(cudaMemcpyAsync(output_cpu, output, sizeof(int) * size, cudaMemcpyDeviceToHost, *cudaStream));
    gpuErrChk(cudaStreamSynchronize(*cudaStream));
    for (size_t i=1; i<size; i++) {
        output_cpu[i] += output_cpu[i-1];
    }
    gpuErrChk(cudaMemcpyAsync(output, output_cpu, sizeof(int) * size, cudaMemcpyHostToDevice, *cudaStream));
    return output_cpu[size-1];
}

// create_sub_batch kernel
template<typename input_t>
__global__ void create_sub_batch(input_t *bin,
                                 size_t *keys,
                                 size_t batch_size,
                                 int *scan,
                                 input_t *bout,
                                 size_t *keys_out,
                                 int n_dest,
                                 int target_node)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    // No need for an explicit cycle: each GPU thread computes this in
    // parallel.
    if (id < batch_size && (keys[id] % n_dest) == target_node) {
        bout[scan[id] - 1] = bin[id];
        keys_out[scan[id] - 1] = keys[id];
    }
}

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
    int *scan_gpu=nullptr;
    int *bout_size=nullptr;

    // constructor
    Emitter(size_t _n_dest,
            size_t _batch_len):
            n_dest(_n_dest),
            batch_len(_batch_len)
    {
        // initialize arrays on GPU
        cudaMalloc(&scan_gpu, sizeof(int) * batch_len);
        // initialize arrays on pinned memory
        cudaMallocHost(&bout_size, sizeof(int));
        // initialize CUDA stream
        gpuErrChk(cudaStreamCreate(&cudaStream));       
    }

    // destructor
    ~Emitter()
    {
        cudaFree(scan_gpu);
        cudaFreeHost(bout_size);
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
        // for each destination connected to this emitter
        for (size_t w=0; w<n_dest; w++) {
            *bout_size = mapped_scan(scan_gpu, b->keys_gpu, b->size, n_dest, w, &cudaStream);
            if (*bout_size > 0) {
                batch_t *b_tosend = new batch_t(*bout_size);
                auto num_of_blocks = ceil(((double) b->size) / threads_per_block);
                create_sub_batch<tuple_t><<<num_of_blocks, threads_per_block, 0, cudaStream>>>(b->data_gpu, b->keys_gpu, b->size, scan_gpu, b_tosend->data_gpu, b_tosend->keys_gpu, n_dest, w);
                assert(cudaGetLastError() == cudaSuccess);
                gpuErrChk(cudaStreamSynchronize(cudaStream));
                this->ff_send_out_to(b_tosend, w);
            }
        }
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
        received += b->size;
        cudaMallocHost(&keys_cpu, sizeof(size_t) * b->size);
        // copy keys_gpu to keys_cpu
        gpuErrChk(cudaMemcpyAsync(keys_cpu, b->keys_gpu, b->size * sizeof(size_t), cudaMemcpyDeviceToHost, cudaStream));
        gpuErrChk(cudaStreamSynchronize(cudaStream));
        // check the correctness of the received batch
        //std::cout << "Worker Keys [ ";
        for (size_t i=0; i<b->size; i++) {
            assert(keys_cpu[i] % par_deg == id);
            //std::cout << keys_cpu[i] << " ";
        }
        //std::cout << "]" << std::endl;
        cudaFreeHost(keys_cpu);
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
