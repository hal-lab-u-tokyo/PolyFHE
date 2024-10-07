SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/codegen/cuda_codegen.cpp \
	hifive/engine/pass/kernel_fusion_pass.cpp \
	hifive/tools/hifive.cpp

HDR=\
	hifive/core/graph/graph.hpp \
	hifive/core/graph/node.hpp \
	hifive/engine/codegen/codegen_base.hpp \
	hifive/engine/codegen/codegen_manager.hpp \
	hifive/engine/codegen/codegen_writer.hpp \
	hifive/engine/codegen/cuda_codegen.hpp \
	hifive/engine/pass/kernel_fusion_pass.hpp

CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -O2 -I./
BIN=build/cc-hifive

$(BIN): $(SRC) $(HDR)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o ./data/graph_poly.png ./data/graph_poly.dot
	dot -Tpng -o ./data/graph_fhe.png ./data/graph_fhe.dot
	dot -Tpng -o ./data/graph_poly_fused_elemwise.png ./data/graph_poly_fused_elemwise.dot
	dot -Tpng -o ./data/graph_poly_fused_alpha.png ./data/graph_poly_fused_alpha.dot
	dot -Tpng -o ./data/graph_poly_fused_beta.png ./data/graph_poly_fused_beta.dot

run: $(BIN)
	./$(BIN)

format:
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666

clean:
	rm -rf $(BIN)