# Author: Gabriele Mencagli
# Date: 23/07/2020

FF_INCLUDES     = $(HOME)/fastflow/

CXX             = g++
CXXFLAGS        = -std=c++14
INCLUDES        = -I $(FF_INCLUDES)
MACRO           = -DFF_BOUNDED_BUFFER -DDEFAULT_BUFFER_CAPACITY=16 -D__RECYCLE__
OPTFLAGS        = -g -O3
LDFLAGS         = -pthread

NVXX            = /usr/local/cuda/bin/nvcc
NVXXFLAGS       = -std=c++14 -x cu 
NVOPTFLAGS      = -w --expt-extended-lambda -O3 -g -gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr
NVOPTFLAGS2      = -w --expt-extended-lambda -O3 -g -gencode arch=compute_53,code=sm_53 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr

# gpu_split is removed for now
all: gpu_map_stateful gpu_map_stateless gpu_filter_stateless gpu_filter_stateful

gpu_map_stateful.o: benchmarks/gpu_map_stateful.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o gpu_map_stateful.o

gpu_map_stateless.o: benchmarks/gpu_map_stateless.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o gpu_map_stateless.o

gpu_filter_stateless.o: benchmarks/gpu_filter_stateless.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o gpu_filter_stateless.o

gpu_filter_stateful.o: benchmarks/gpu_filter_stateful.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o gpu_filter_stateful.o

# gpu_split.o: benchmarks/gpu_split.cpp
# 	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o gpu_split.o

gpu_map_stateful: gpu_map_stateful.o
	$(NVXX) gpu_map_stateful.o -o gpu_map_stateful

gpu_map_stateless: gpu_map_stateless.o
	$(NVXX) gpu_map_stateless.o -o gpu_map_stateless

gpu_filter_stateless: gpu_filter_stateless.o
	$(NVXX) gpu_filter_stateless.o -o gpu_filter_stateless

gpu_filter_stateful: gpu_filter_stateful.o
	$(NVXX) gpu_filter_stateful.o -o gpu_filter_stateful

# gpu_split: gpu_split.o
# 	$(NVXX) gpu_split.o -o gpu_split

clean:
	rm -f gpu_map_stateful
	rm -f gpu_map_stateless
	rm -f gpu_filter_stateless
	rm -f gpu_filter_stateful
	rm -f gpu_split
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean