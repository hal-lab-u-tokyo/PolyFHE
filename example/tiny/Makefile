SRC=example.cu
BIN=example.out


$(BIN): $(SRC) 
	nvcc -o $(BIN) $(SRC)
	./$(BIN)

opt: $(BIN)
	./$(BIN) 1

noopt: $(BIN)
	./$(BIN) 0

clean:
	rm -rf $(BIN_EXAMPLE)