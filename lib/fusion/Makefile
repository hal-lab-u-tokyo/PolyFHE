SRC=\
	fhe.cc \
	poly.cc \
	main.cc

CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -O3
BIN=fusion

all: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o ./graph/graph_poly.png ./graph/graph_poly.dot
	dot -Tpng -o ./graph/graph_fhe.png ./graph/graph_fhe.dot

run: 
	./$(BIN)
	make dot