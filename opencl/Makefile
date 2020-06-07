CC=g++

CFLAGS= -std=c++11 -Wall -DUNIX -g -DDEBUG

DIFFCAL_DIR := /home/haakon/Dropbox/Documents/NTNU/TDT4900/dataconstruction/denoise_util/diffcal

# Check for 32-bit vs 64-bit
PROC_TYPE = $(strip $(shell uname -m | grep 64))

# Check for Mac OS
OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))

# MacOS System
ifneq ($(DARWIN),)
	CFLAGS += -DMAC
	LIBS=-framework OpenCL

	ifeq ($(PROC_TYPE),)
		CFLAGS+=-arch i386
	else
		CFLAGS+=-arch x86_64
	endif
else

# Linux OS
LIBS=-lOpenCL -lGL -lOpenImageIO -fopenmp
ifeq ($(PROC_TYPE),)
	CFLAGS+=-m32
else
	CFLAGS+=-m64
endif

# Check for Linux-AMD
ifdef AMDAPPSDKROOT
	INC_DIRS=. $(AMDAPPSDKROOT)/include
	ifeq ($(PROC_TYPE),)
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86
	else
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86_64
	endif
else

# Check for Linux-Nvidia
ifdef NVSDKCOMPUTE_ROOT
	INC_DIRS=. $(NVSDKCOMPUTE_ROOT)/OpenCL/common/inc
endif

endif
endif

bmfr: bmfr.cpp CLUtils/CLUtils.hpp CLUtils/CLUtils.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

bmfr-debug: bmfr.cpp CLUtils/CLUtils.hpp CLUtils/CLUtils.cpp 
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS) -D WITH_VISBUF

bmfr-evaluate: bmfr.cpp CLUtils/CLUtils.hpp CLUtils/CLUtils.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS) -D EVALUATION_MODE -I $(DIFFCAL_DIR) -l diffcal -L $(DIFFCAL_DIR) -lvmaf

.PHONY: clean

clean:
	rm -f bmfr
