ROOT=../../..
CXXFLAGS = -std=c++17 -O3 -I. -I$(ROOT)/thirdparty -I$(ROOT)/thirdparty/phantom-fhe/include -I$(ROOT)/
CXXFLAGS += -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75
LIBPATH = $(ROOT)/build/thirdparty/phantom-fhe/lib
LDFLAGS = -Xlinker -rpath -Xlinker $(LIBPATH) -L$(LIBPATH) -lPhantom -lpthread -lnvToolsExt

BIN_COMPILER = $(ROOT)/build/polyfhe-cc
BIN_EXAMPLE = build/example.out
COMPILER_OUT = build/generated.cu
DOT = output.dot
CONFIG = config.csv
N_OPT = 1

all: $(BIN_EXAMPLE)
	@echo "Running example..."
	@mkdir -p build
	./$(BIN_EXAMPLE) $(N_OPT)

noopt: generated-noopt.cu example.cu
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o build/example-noopt.out generated-noopt.cu example.cu $(ROOT)/polyfhe/kernel/polynomial.cu
	./build/example-noopt.out 1

 
reg: generated-reg.cu example.cu
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o build/example-reg.out generated-reg.cu example.cu $(ROOT)/polyfhe/kernel/polynomial.cu
	./build/example-reg.out 1

l2: generated-l2.cu example.cu
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o build/example-l2.out generated-l2.cu example.cu $(ROOT)/polyfhe/kernel/polynomial.cu
	./build/example-l2.out 3

shared: generated-shared.cu example.cu
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o build/example-shared.out generated-shared.cu example.cu $(ROOT)/polyfhe/kernel/polynomial.cu
	./build/example-shared.out 3

nsys: $(BIN_EXAMPLE)
	nsys start -c cudaProfilerApi
	nsys launch -w true ./$(BIN_EXAMPLE)

ncu: $(BIN_EXAMPLE)
	ncu --nvtx --nvtx-include "compute/" ./$(BIN_EXAMPLE) 1

dot:
	find ./build -type f -name "*.dot" -exec sh -c 'for f; do dot -Tpng "$$f" -o "$${f%.dot}.png"; done' _ {} +
	find ./       -type f -name "*.dot" -exec sh -c 'for f; do dot -Tpng "$$f" -o "$${f%.dot}.png"; done' _ {} +
	
clean:
	rm -rf build