# Author: Gabriele Mencagli
# Date: 23/07/2020

FF_INCLUDES     = $(HOME)/fastflow/

CXX             = g++
CXXFLAGS        = -std=c++14
INCLUDES        = -I $(FF_INCLUDES)
MACRO           = -DFF_BOUNDED_BUFFER -DDEFAULT_BUFFER_CAPACITY=16 -D__RECYCLE__ -DNDEBUG
OPTFLAGS        = -g -O3
LDFLAGS         = -pthread

NVXX            = /usr/local/cuda/bin/nvcc
NVXXFLAGS       = -std=c++14 -x cu --compiler-options "-Wall -Wextra -Wpedantic"
NVOPTFLAGS      = -w --expt-extended-lambda -O3 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr

ARCH = $(shell arch)
ifeq ($(ARCH), x86_64)
	NVOPTFLAGS := $(NVOPTFLAGS) -gencode arch=compute_35,code=sm_35
endif
ifeq ($(ARCH), aarch64)
	NVOPTFLAGS := $(NVOPTFLAGS) -gencode arch=compute_53,code=sm_53
endif

debug: MACRO := $(MACRO) -UNDEBUG
debug: OPTFLAGS := $(OPTFLAGS) -gtoggle
debug: all

# gpu_split is removed for now
all: cpu_map_stateful gpu_map_stateful gpu_map_stateless gpu_map_stateless_one_per_warp gpu_filter_stateless gpu_filter_stateful gpu_map_stateful_one_per_warp gpu_map_stateful_no_warps gpu_filter_stateful_one_per_warp gpu_filter_stateful_no_warps

cpu_map_stateful.o: benchmarks/cpu_map_stateful.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o cpu_map_stateful.o

gpu_map_stateful.o: benchmarks/gpu_map_stateful.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_map_stateful.o

gpu_map_stateless.o: benchmarks/gpu_map_stateless.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_map_stateless.o

gpu_map_stateless_one_per_warp.o: benchmarks/gpu_map_stateless_one_per_warp.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_map_stateless_one_per_warp.o

gpu_filter_stateless.o: benchmarks/gpu_filter_stateless.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_filter_stateless.o

gpu_filter_stateful.o: benchmarks/gpu_filter_stateful.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_filter_stateful.o

gpu_map_stateful_one_per_warp.o: benchmarks/gpu_map_stateful_one_per_warp.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_map_stateful_one_per_warp.o

gpu_map_stateful_no_warps.o: benchmarks/gpu_map_stateful_no_warps.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_map_stateful_no_warps.o

gpu_filter_stateful_one_per_warp.o: benchmarks/gpu_filter_stateful_one_per_warp.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_filter_stateful_one_per_warp.o

gpu_filter_stateful_no_warps.o: benchmarks/gpu_filter_stateful_no_warps.cpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_filter_stateful_no_warps.o

# gpu_split.o: benchmarks/gpu_split.cpp
# 	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) $(USER_DEFINES) -c $< -o gpu_split.o

cpu_map_stateful: cpu_map_stateful.o
	$(NVXX) cpu_map_stateful.o -o cpu_map_stateful

gpu_map_stateful: gpu_map_stateful.o
	$(NVXX) gpu_map_stateful.o -o gpu_map_stateful

gpu_map_stateless: gpu_map_stateless.o
	$(NVXX) gpu_map_stateless.o -o gpu_map_stateless

gpu_map_stateless_one_per_warp: gpu_map_stateless_one_per_warp.o
	$(NVXX) gpu_map_stateless_one_per_warp.o -o gpu_map_stateless_one_per_warp

gpu_filter_stateless: gpu_filter_stateless.o
	$(NVXX) gpu_filter_stateless.o -o gpu_filter_stateless

gpu_filter_stateful: gpu_filter_stateful.o
	$(NVXX) gpu_filter_stateful.o -o gpu_filter_stateful

gpu_map_stateful_one_per_warp: gpu_map_stateful_one_per_warp.o
	$(NVXX) gpu_map_stateful_one_per_warp.o -o gpu_map_stateful_one_per_warp

gpu_map_stateful_no_warps: gpu_map_stateful_no_warps.o
	$(NVXX) gpu_map_stateful_no_warps.o -o gpu_map_stateful_no_warps

gpu_filter_stateful_one_per_warp: gpu_filter_stateful_one_per_warp.o
	$(NVXX) gpu_filter_stateful_one_per_warp.o -o gpu_filter_stateful_one_per_warp

gpu_filter_stateful_no_warps: gpu_filter_stateful_no_warps.o
	$(NVXX) gpu_filter_stateful_no_warps.o -o gpu_filter_stateful_no_warps

# gpu_split: gpu_split.o
# 	$(NVXX) gpu_split.o -o gpu_split

clean:
	rm -f gpu_map_stateful*
	rm -f gpu_map_stateless*
	rm -f gpu_filter_stateless*
	rm -f gpu_filter_stateful*
	rm -f gpu_split
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
