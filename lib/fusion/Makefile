SRC=\
	fhe.cc \
	poly.cc \
	main.cc

CXXFLAGS=-std=c++11 -Wall -Wextra -pedantic -O3
BIN=fusion

all: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o graph_poly.png graph_poly.dot
	dot -Tpng -o graph_fhe.png graph_fhe.dot

run: 
	./$(BIN)
	make dot