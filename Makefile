SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/codegen/cuda_codegen.cpp \
	hifive/engine/pass/analyze_intra_node_pass.cpp \
	hifive/engine/pass/calculate_memory_traffic_pass.cpp \
	hifive/engine/pass/data_reuse_pass.cpp \
	hifive/engine/pass/extract_subgraph_pass.cpp \
	hifive/engine/pass/lowering_ckks_to_poly_pass.cpp \
	hifive/engine/pass/rewrite_ntt_pass.cpp \
	hifive/engine/pass/set_block_phase_pass.cpp \
	hifive/frontend/exporter.cpp \
	hifive/frontend/parser.cpp \
	hifive/tools/hifive.cpp

SRC_RUNTIME=\
	hifive/kernel/device_context.cu \
	hifive/kernel/polynomial.cu \
	hifive/kernel/ntt.cu

SRC_TEST=\
	test/test_ntt.cu \
	test/main.cu \
	test/util.cu

HDR=\
	hifive/core/graph/edge.hpp \
	hifive/core/graph/graph.hpp \
	hifive/core/graph/node.hpp \
	hifive/core/logger.hpp \
	hifive/core/param.hpp \
	hifive/engine/codegen/codegen_base.hpp \
	hifive/engine/codegen/codegen_manager.hpp \
	hifive/engine/codegen/codegen_writer.hpp \
	hifive/engine/codegen/cuda_codegen.hpp \
	hifive/engine/pass/analyze_intra_node_pass.hpp \
	hifive/engine/pass/calculate_memory_traffic_pass.hpp \
	hifive/engine/pass/data_reuse_pass.hpp \
	hifive/engine/pass/extract_subgraph_pass.hpp \
	hifive/engine/pass/lowering_ckks_to_poly_pass.hpp \
	hifive/engine/pass/rewrite_ntt_pass.hpp \
	hifive/engine/pass/set_block_phase_pass.hpp \
	hifive/frontend/exporter.hpp \
	hifive/frontend/parser.hpp

OBJ=$(SRC:.cpp=.o)

CXXFLAGS=-g -std=c++2a -Wall -Wextra -pedantic -O2 -I./
LDFLAGS=-lboost_graph -lboost_program_options
BIN=build/cc-hifive

CXXFLAGS_RUNTIME=-g -std=c++17 -O2 -I./hifive/kernel/FullRNS-HEAAN/src/ -I./  --relocatable-device-code true
LDFLAGS_RUNTIME=-L./hifive/kernel/FullRNS-HEAAN/lib/ -lFRNSHEAAN
BIN_RUNTIME=build/bench

$(BIN): $(SRC) $(HDR) $(OBJ)
	rm -rf ./build
	mkdir -p build
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(SRC_TEST) $(SRC_RUNTIME) $(SRC) $(HDR)
	mkdir -p build
	nvcc -o build/test $(SRC_TEST) $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/test

TARGET=data/graph_poly.dot

run: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i $(TARGET) -p 
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-opt
	mv build build-opt
	
run-noopt: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i $(TARGET) --noopt -p
	make dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	rm -rf build-noopt
	mv build build-noopt

fhe: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i data/graph_fhe.dot --noopt
	make dot

bin:
	nvcc -o build/bench build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/bench

bin-noopt:
	nvcc -o build-noopt/bench build-noopt/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build-noopt/bench

dot:
	find ./build -iname *.dot -exec dot -Tpng -o {}.png {} \;
	find ./data -iname *.dot -exec dot -Tpng -o {}.png {} \;

format:
	find ./hifive ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

clean:
	rm -rf build
	rm -rf $(BIN) $(OBJ)