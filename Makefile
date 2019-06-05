CC=g++
# OMPFLAG=
OMPFLAG=-fopenmp
USEATEN=-DUSEATEN
# USEATEN=

CXXFLAGS=-std=c++11 -O3 -fPIC -Wall $(OMPFLAG)

ATEN_MACRO=-D_GLIBCXX_USE_CXX11_ABI=0
ATEN_INCLUDE_FLAGS=-I$(shell python -c \
				   "import torch.utils.cpp_extension as cpp_extension; \
				   print(cpp_extension.include_paths()[0])")
ATEN_CXXFLAGS=$(ATEN_MACRO) $(ATEN_INCLUDE_FLAGS)

ATEN_LIBRARY_PATH=$(shell python -c \
				  "import os,torch; \
				  print(os.path.dirname(torch.__file__)+'/lib')")
ANACONDALIBROOT=$(shell python -c \
				"import sysconfig; \
				print(sysconfig.get_config_vars()['LIBDIR'])")
ATEN_LDFLAGS=-L$(ATEN_LIBRARY_PATH) -L$(ANACONDALIBROOT) -lcaffe2 -lc10 \
			 -Wl,--no-as-needed -Wl,-rpath=$(ATEN_LIBRARY_PATH) -Wl,-rpath=$(ANACONDALIBROOT) 


BINNAME=main
.PHONY:all obj bin clean
all:obj bin
obj:
ifeq ($(USEATEN), -DUSEATEN)
	$(CC) -c test.cpp -o test.o $(CXXFLAGS) $(USEATEN) $(ATEN_CXXFLAGS)
else
	$(CC) -c test.cpp -o test.o $(CXXFLAGS)
endif
bin:obj
ifeq ($(USEATEN), -DUSEATEN)
	$(CC) test.o -o $(BINNAME) $(OMPFLAG) $(ATEN_LDFLAGS)
else
	$(CC) test.o -o $(BINNAME) $(OMPFLAG)
endif
clean:
	-rm -rf $(BINNAME) test.o
	-rm -rf log [0-4]
