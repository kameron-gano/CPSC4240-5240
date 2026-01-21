CXX := g++

CXXFLAGS := -std=c++17 -O3 -Xpreprocessor -fopenmp -lomp -pthread  -I.

LIBOMP_ROOT := /opt/homebrew/opt/libomp
CXXFLAGS += -I$(LIBOMP_ROOT)/include
LDFLAGS := -L$(LIBOMP_ROOT)/lib 

all: matrixmult

matrixmult: matrixmult.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f matrixmult

.PHONY: all clean