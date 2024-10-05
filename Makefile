SRC=\
	hifive/core/graph/graph.cpp \
	hifive/core/graph/node.cpp \
	hifive/engine/pass/kernel_fusion_pass.cpp \
	hifive/tools/hifive.cpp

CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -O2 -I./
BIN=cc-hifive

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o ./data/graph_poly.png ./data/graph_poly.dot
	dot -Tpng -o ./data/graph_fhe.png ./data/graph_fhe.dot
	dot -Tpng -o ./data/graph_poly_fused_elemwise.png ./data/graph_poly_fused_elemwise.dot
	dot -Tpng -o ./data/graph_poly_fused_alpha.png ./data/graph_poly_fused_alpha.dot
	dot -Tpng -o ./data/graph_poly_fused_beta.png ./data/graph_poly_fused_beta.dot

run: $(BIN)
	./$(BIN)
	make dot

format:
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
	find ./hifive -iname *.hpp -o -iname *.cpp -o -iname *.cu | xargs chmod 666