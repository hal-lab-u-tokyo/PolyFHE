ROOT=../..
CXXFLAGS = -std=c++17 -O3 -I. -I$(ROOT)/thirdparty -I$(ROOT)/thirdparty/phantom-fhe/include -I$(ROOT)/
CXXFLAGS += -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75
LIBPATH = $(ROOT)/build/thirdparty/phantom-fhe/lib
LDFLAGS = -Xlinker -rpath -Xlinker $(LIBPATH) -L$(LIBPATH) -lPhantom -lpthread -lnvToolsExt

BIN_COMPILER = $(ROOT)/build/polyfhe-cc
BIN_EXAMPLE = build/example.out
COMPILER_OUT = build/generated.cu
DOT = output.dot
CONFIG = config.csv

all: $(BIN_EXAMPLE)
	@echo "Running example..."
	@mkdir -p build
	./$(BIN_EXAMPLE) 1

$(BIN_EXAMPLE): $(COMPILER_OUT) example.cu
	@echo "Compiling example code..."
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o $@ $(COMPILER_OUT) example.cu $(ROOT)/polyfhe/kernel/poly_global.cu 

$(COMPILER_OUT): $(DOT)
	@echo "Generating GPU code using PolyFHE..."
	@mkdir -p build
	$(BIN_COMPILER) -i $(DOT) -p -c $(CONFIG) $(OPT_LEVEL)

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