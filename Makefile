SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/codegen/cuda_codegen.cpp \
	hifive/engine/pass/kernel_fusion_pass.cpp \
	hifive/frontend/parser.cpp \
	hifive/tools/hifive.cpp

HDR=\
	hifive/core/graph/graph.hpp \
	hifive/core/graph/node.hpp \
	hifive/engine/codegen/codegen_base.hpp \
	hifive/engine/codegen/codegen_manager.hpp \
	hifive/engine/codegen/codegen_writer.hpp \
	hifive/engine/codegen/cuda_codegen.hpp \
	hifive/engine/pass/kernel_fusion_pass.hpp \
	hifive/frontend/parser.hpp
OBJ=$(SRC:.cpp=.o)

CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -O2 -I./
LDFLAGS=-lboost_graph
BIN=build/cc-hifive

$(BIN): $(SRC) $(HDR) $(OBJ)
	mkdir -p build
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dot:
	dot -Tpng -o ./data/graph_fhe.png ./data/graph_fhe.dot

run: $(BIN)
	./$(BIN) ./data/graph_fhe.dot
	nvcc -o build/gen_cuda build/gen_cuda_main.cu
	./build/gen_cuda

format:
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

clean:
	rm -rf $(BIN)