SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/codegen/cuda_codegen.cpp \
	hifive/engine/pass/calculate_memory_traffic_pass.cpp \
	hifive/engine/pass/kernel_fusion_pass.cpp \
	hifive/frontend/exporter.cpp \
	hifive/frontend/parser.cpp \
	hifive/tools/hifive.cpp

SRC_RUNTIME=\
	hifive/kernel/device_context.cu \
	hifive/kernel/polynomial.cu

SRC_TEST=\
	test/test_poly.cu \
	test/main.cu \
	test/util.cu

HDR=\
	hifive/core/graph/graph.hpp \
	hifive/core/graph/node.hpp \
	hifive/core/logger.hpp \
	hifive/engine/codegen/codegen_base.hpp \
	hifive/engine/codegen/codegen_manager.hpp \
	hifive/engine/codegen/codegen_writer.hpp \
	hifive/engine/codegen/cuda_codegen.hpp \
	hifive/engine/pass/calculate_memory_traffic_pass.hpp \
	hifive/engine/pass/kernel_fusion_pass.hpp \
	hifive/frontend/exporter.hpp \
	hifive/frontend/parser.hpp

OBJ=$(SRC:.cpp=.o)

CXXFLAGS=-g -std=c++2a -Wall -Wextra -pedantic -O2 -I./
LDFLAGS=-lboost_graph -lboost_program_options
BIN=build/cc-hifive

CXXFLAGS_RUNTIME=-g -std=c++17 -O2 -I./hifive/kernel/FullRNS-HEAAN/src/ -I./  --relocatable-device-code true
LDFLAGS_RUNTIME=-L./hifive/kernel/FullRNS-HEAAN/lib/ -lFRNSHEAAN
BIN_RUNTIME=build/gen_cuda

$(BIN): $(SRC) $(HDR) $(OBJ)
	rm -rf ./build
	mkdir -p build
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(SRC_TEST) $(SRC_RUNTIME)
	mkdir -p build
	nvcc -o build/test $(SRC_TEST) $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./build/test

run: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i ./data/graph_poly.dot -o
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	make dot

run-noopt: $(BIN)
	rm -f ./build/*.dot
	rm -f ./build/*.png
	./$(BIN) -i ./data/graph_poly.dot
	nvcc -o $(BIN_RUNTIME) build/generated.cu $(SRC_RUNTIME) $(CXXFLAGS_RUNTIME) $(LDFLAGS_RUNTIME)
	./$(BIN_RUNTIME)
	make dot

dot:
	dot -Tpng -o ./data/graph_poly.png ./data/graph_poly.dot
	find ./build -iname *.dot -exec dot -Tpng -o {}.png {} \;

format:
	find ./hifive ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive ./test -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

clean:
	rm -rf build
	rm -rf $(BIN) $(OBJ)