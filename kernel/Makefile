CXXFLAGS = -std=c++11 -O3

SRC=\
	Ciphertext.cpp \
	Context.cpp \
	EvaluatorUtils.cpp \
	Key.cpp \
	Numb.cpp \
	Plaintext.cpp \
	Scheme.cpp \
	SchemeAlgo.cpp \
	SecretKey.cpp \
	StringUtils.cpp \
	TestScheme.cpp \
	TimeUtils.cpp \
	main.cpp

all: $(SRC)
	$(CXX) $(CXXFLAGS) -o TestScheme $(SRC)

POLY_SRC=\
	polynomial.cu \
	polynomial_test.cu
POLY_HDR=\
	polynomial.h
POLY_OBJS=$(POLY_SRC:.cu=.o)

cuda: $(POLY_SRC)
	nvcc -std=c++11 -O3 $(POLY_SRC) --device-c
	nvcc $(POLY_OBJS) -o test_polynomial

format: $(POLY_SRC) $(POLY_HDR)
	clang-format -i $^ | chmod 777 $^