ROOT=../..
CXXFLAGS = -std=c++17 -O3 -I. -I$(ROOT)/thirdparty -I$(ROOT)/thirdparty/phantom-fhe/include -I$(ROOT)/
LIBPATH = $(ROOT)/build/thirdparty/phantom-fhe/lib
LDFLAGS = -Xlinker -rpath -Xlinker $(LIBPATH) -L$(LIBPATH) -lPhantom -lpthread

BIN_COMPILER = $(ROOT)/build/polyfhe-cc
BIN_EXAMPLE = build/example.out
COMPILER_OUT = build/generated.cu
DOT = graph_poly.dot
CONFIG = config.csv

all: $(BIN_EXAMPLE)
	@echo "Running example..."
	@mkdir -p build
	./$(BIN_EXAMPLE)

$(BIN_EXAMPLE): $(COMPILER_OUT) example.cu
	@echo "Compiling example code..."
	@mkdir -p build
	nvcc $(CXXFLAGS) $(LDFLAGS) -o $@ $(COMPILER_OUT) example.cu 

$(COMPILER_OUT):
	@echo "Generating GPU code using PolyFHE..."
	@mkdir -p build
	$(BIN_COMPILER) -i $(DOT) -p -c $(CONFIG)

clean:
	rm -rf build