SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/codegen/cuda_codegen.cpp \
	hifive/engine/pass/calculate_memory_traffic_pass.cpp \
	hifive/engine/pass/kernel_fusion_pass.cpp \
	hifive/frontend/exporter.cpp \
	hifive/frontend/parser.cpp \
	hifive/tools/hifive.cpp

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

$(BIN): $(SRC) $(HDR) $(OBJ)
	mkdir -p build
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dot:
	dot -Tpng -o ./data/graph_poly.png ./data/graph_poly.dot
	find ./build -iname *.dot -exec dot -Tpng -o {}.png {} \;

run: $(BIN)
	./$(BIN) -i ./data/graph_poly.dot
	nvcc -o build/gen_cuda build/gen_cuda_main.cu
	./build/gen_cuda
	make dot

format:
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

clean:
	rm -rf $(BIN) $(OBJ)