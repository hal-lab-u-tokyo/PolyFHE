SRC=\
	src/fhe-op.cc \
	src/poly-op.cc \
	src/main.cc

CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -O2
BIN=fusion

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o ./data/graph_poly.png ./data/graph_poly.dot
	dot -Tpng -o ./data/graph_fhe.png ./data/graph_fhe.dot
	dot -Tpng -o ./data/graph_poly_fused_elemwise.png ./data/graph_poly_fused_elemwise.dot

run: $(BIN)
	./$(BIN)
	make dot