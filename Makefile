CC=g++
# OMPFLAG=
OMPFLAG=-fopenmp
# USEATEN=-DUSEATEN
USEATEN=

PYTHONVER=3.6
ANACONDA_DIR=/opt/software/anaconda/3
ANACONDALIBROOT=$(ANACONDA_DIR)/lib
PYTORCHROOT=$(ANACONDALIBROOT)/python$(PYTHONVER)/site-packages/torch
ATEN_MACRO=-D_GLIBCXX_USE_CXX11_ABI=0
ATEN_INCLUDE_FLAGS=-I$(PYTORCHROOT)/lib/include

ATEN_CXXFLAGS=$(ATEN_MACRO) $(ATEN_INCLUDE_FLAGS)
ATEN_LIBRARY_FLAGS=-L$(PYTORCHROOT)/lib -L$(ANACONDALIBROOT) -lcaffe2 -lc10
ATEN_LDFLAGS=-Wl,--no-as-needed -Wl,-rpath=$(PYTORCHROOT)/lib -Wl,-rpath=$(ANACONDALIBROOT) 

CXXFLAGS=-std=c++11 -O3 -fPIC $(OMPFLAG)

BINNAME=main
.PHONY:all obj bin clean
all:obj bin
obj:
ifeq ($(USEATEN), -DUSEATEN)
	g++ -c test.cpp -o test.o $(CXXFLAGS) $(USEATEN) $(ATEN_CXXFLAGS)
else
	g++ -c test.cpp -o test.o $(CXXFLAGS)
endif
bin:obj
ifeq ($(USEATEN), -DUSEATEN)
	g++ test.o -o $(BINNAME) $(OMPFLAG) $(ATEN_LDFLAGS) $(ATEN_LIBRARY_FLAGS)
else
	g++ test.o -o $(BINNAME) $(OMPFLAG)
endif
clean:
	-rm -rf $(BINNAME) test.o
