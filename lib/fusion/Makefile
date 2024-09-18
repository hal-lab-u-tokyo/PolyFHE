SRC=\
	poly.cc \
	main.cc

CXXFLAGS=-std=c++11 -Wall -Wextra -pedantic -O3
BIN=fusion

all: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

dot:
	dot -Tpng -o graph.png graph.dot

run: 
	./$(BIN)
	make dot