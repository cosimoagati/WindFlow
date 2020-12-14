# Author: Gabriele Mencagli
# Date: 23/07/2020

FF_INCLUDES		= $(HOME)/fastflow/

CXX 			= g++
CXXFLAGS		= -std=c++14
INCLUDES		= -I $(FF_INCLUDES)
MACRO                   = -DFF_BOUNDED_BUFFER -DNDEBUG
OPTFLAGS		= -g -O3
LDFLAGS			= -pthread

NVXX 			= /usr/local/cuda/bin/nvcc
NVXXFLAGS               = -std=c++14 -x cu --compiler-options "-Wall -Wextra -Wpedantic"
NVOPTFLAGS              = -w --expt-extended-lambda -O3 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr

ARCH = $(shell arch)
ifeq ($(ARCH), x86_64)
	NVOPTFLAGS := $(NVOPTFLAGS) -gencode arch=compute_53,code=sm_53
endif
ifeq ($(ARCH), aarch64)
	NVOPTFLAGS := $(NVOPTFLAGS) -gencode arch=compute_53,code=sm_53
endif

debug: MACRO := $(MACRO) -UNDEBUG
debug: OPTFLAGS := $(OPTFLAGS) -gtoggle
debug: all

all: test_scatter_v1 test_scatter_v2 test_scatter_v3 test_scatter_v4 test_stateful_v1 test_stateful_v2 test_stateful_v3 test_stateful_v4 test_stateful_v5 test_stateful_v6

test_scatter_v1.o: scattering/test_v1.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS)  -c $< -o test_scatter_v1.o

test_scatter_v2.o: scattering/test_v2.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_scatter_v2.o

test_scatter_v3.o: scattering/test_v3.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_scatter_v3.o

test_scatter_v4.o: scattering/test_v4.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_scatter_v4.o

stateful_common.o: stateful/common.cpp stateful/common.hpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o stateful_common.o

test_stateful_v1.o: stateful/test_v1.cpp stateful/common.hpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v1.o

test_stateful_v2.o: stateful/test_v2.cpp stateful/common.hpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v2.o

test_stateful_v3.o: stateful/test_v3.cpp stateful/common.hpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v3.o

test_stateful_v4.o: stateful/test_v4.cpp stateful/common.hpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v4.o

test_stateful_v5.o: stateful/test_v5.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v5.o

test_stateful_v6.o: stateful/test_v6.cpp zipf.hpp
	$(NVXX) $(NVXXFLAGS) $(INCLUDES) $(MACRO) $(NVOPTFLAGS) -c $< -o test_stateful_v6.o

test_scatter_v1: test_scatter_v1.o
	$(NVXX) test_scatter_v1.o -o test_scatter_v1

test_scatter_v2: test_scatter_v2.o
	$(NVXX) test_scatter_v2.o -o test_scatter_v2

test_scatter_v3: test_scatter_v3.o
	$(NVXX) test_scatter_v3.o -o test_scatter_v3

test_scatter_v4: test_scatter_v4.o
	$(NVXX) test_scatter_v4.o -o test_scatter_v4

test_stateful_v1: test_stateful_v1.o stateful_common.o
	$(NVXX) test_stateful_v1.o stateful_common.o -o test_stateful_v1

test_stateful_v2: test_stateful_v2.o stateful_common.o
	$(NVXX) test_stateful_v2.o stateful_common.o -o test_stateful_v2

test_stateful_v3: test_stateful_v3.o stateful_common.o
	$(NVXX) test_stateful_v3.o stateful_common.o -o test_stateful_v3

test_stateful_v4: test_stateful_v4.o stateful_common.o
	$(NVXX) test_stateful_v4.o stateful_common.o -o test_stateful_v4

test_stateful_v5: test_stateful_v5.o
	$(NVXX) test_stateful_v5.o -o test_stateful_v5

test_stateful_v6: test_stateful_v6.o
	$(NVXX) test_stateful_v6.o -o test_stateful_v6

clean:
	rm -f test_scatter_v1
	rm -f test_scatter_v2
	rm -f test_scatter_v3
	rm -f test_scatter_v4
	rm -f test_stateful_v1
	rm -f test_stateful_v2
	rm -f test_stateful_v3
	rm -f test_stateful_v4
	rm -f test_stateful_v5
	rm -f test_stateful_v6
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
