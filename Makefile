SRC=\
	polyfhe/core/graph/graph.cpp \
	polyfhe/core/graph/node.cpp \
	polyfhe/core/config.cpp \
	polyfhe/engine/codegen/cuda_codegen.cpp \
	polyfhe/engine/pass/analyze_intra_node_pass.cpp \
	polyfhe/engine/pass/calculate_memory_traffic_pass.cpp \
	polyfhe/engine/pass/calculate_smem_size_pass.cpp \
	polyfhe/engine/pass/data_reuse_pass.cpp \
	polyfhe/engine/pass/extract_subgraph_pass.cpp \
	polyfhe/engine/pass/kernel_launch_config_pass.cpp \
	polyfhe/engine/pass/lowering_ckks_to_poly_pass.cpp \
	polyfhe/engine/pass/rewrite_ntt_pass.cpp \
	polyfhe/engine/pass/set_block_phase_pass.cpp \
	polyfhe/frontend/exporter.cpp \
	polyfhe/frontend/parser.cpp \
	polyfhe/tools/polyfhe.cpp \
	polyfhe/utils.cpp

SRC_RUNTIME=\
	polyfhe/kernel/device_context.cu \
	polyfhe/kernel/polynomial.cpp \
	polyfhe/utils.cpp
	
SRC_TEST=\
	test/test_ntt.cu \
	test/main.cu \
	test/util.cu

HDR=\
	polyfhe/core/graph/edge.hpp \
	polyfhe/core/graph/graph.hpp \
	polyfhe/core/graph/node.hpp \
	polyfhe/core/logger.hpp \
	polyfhe/core/config.hpp \
	polyfhe/engine/codegen/codegen_base.hpp \
	polyfhe/engine/codegen/codegen_manager.hpp \
	polyfhe/engine/codegen/codegen_writer.hpp \
	polyfhe/engine/codegen/cuda_codegen.hpp \
	polyfhe/engine/pass/analyze_intra_node_pass.hpp \
	polyfhe/engine/pass/calculate_memory_traffic_pass.hpp \
	polyfhe/engine/pass/calculate_smem_size_pass.hpp \
	polyfhe/engine/pass/data_reuse_pass.hpp \
	polyfhe/engine/pass/extract_subgraph_pass.hpp \
	polyfhe/engine/pass/kernel_launch_config_pass.hpp \
	polyfhe/engine/pass/lowering_ckks_to_poly_pass.hpp \
	polyfhe/engine/pass/pass_base.hpp \
	polyfhe/engine/pass/pass_manager.hpp \
	polyfhe/engine/pass/rewrite_ntt_pass.hpp \
	polyfhe/engine/pass/set_block_phase_pass.hpp \
	polyfhe/frontend/exporter.hpp \
	polyfhe/frontend/parser.hpp \
	polyfhe/utils.hpp

OBJ=$(SRC:.cpp=.o)

CXXFLAGS=-g -std=c++2a -Wall -Wextra -pedantic -O2 -I./
LDFLAGS=-lboost_graph -lboost_program_options
BIN=build/cc-polyfhe

CXXFLAGS_RUNTIME=-g -std=c++17 -I./  --relocatable-device-code true
LDFLAGS_RUNTIME=
BIN_RUNTIME=build/bench

$(BIN): $(SRC) $(HDR) $(OBJ)
	rm -rf ./build
	mkdir -p build
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

%.o: %.cpp %.hpp $(HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(SRC_TEST) $(SRC_RUNTIME) $(SRC) $(HDR)
	mkdir -p build
	nvcc -o build/test $(SRC_TEST) $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/test

TARGET=data/graph_poly.dot
CONFIG=config.csv

run: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i $(TARGET) -p -c $(CONFIG)
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-opt
	mv build build-opt
	
run-noopt: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i $(TARGET) --noopt -p -c $(CONFIG)
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-noopt
	mv build build-noopt

run-fhe: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i data/graph_fhe.dot -c $(CONFIG)
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-opt
	mv build build-opt

run-fhe-noopt: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i data/graph_fhe.dot --noopt -c $(CONFIG)
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-noopt
	mv build build-noopt

bin:
	nvcc -o build/bench build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/bench

bin-opt:
	nvcc -o build-opt/bench build-opt/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build-opt/bench

bin-noopt:
	nvcc -o build-noopt/bench build-noopt/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build-noopt/bench

dot:
	find ./build -iname *.dot -exec dot -Tpng -o {}.png {} \;
	find ./data -iname *.dot -exec dot -Tpng -o {}.png {} \;

format:
	find ./polyfhe ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./polyfhe ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

estimate:
	nvcc -o build/estimate polyfhe/estimator/dram.cu $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/estimate > data/dram_latency.csv
	python3 data/plot_dram_latency.py

estimate-ptx:
	nvcc --ptx -o build/estimate.ptx polyfhe/estimator/dram.cu $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)

clean:
	rm -rf build
	rm -rf $(BIN) $(OBJ)