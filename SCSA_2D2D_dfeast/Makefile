include make.inc

# include and lib paths
INCLUDES= -I$(FEAST_ROOT)/include
LIB_PATH=  -L$(FEAST_ROOT)/lib/x86_64

# libraries to link against
#LIBS= -lnvblas -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lcudart -lcublas -lcholinv -lstdc++ -liomp5 -lpthread -lm
#LIBS= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lcudart -lcublas -lcholinv -lstdc++ -liomp5 -lpthread -lm
#LIBS= -mkl=parallel
LIBS= -lfeast -lfeast_sparse -mkl=parallel #-lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
#LIBS+= -lstdc++ -liomp5 -lpthread -lm

CPP_SRC= SCSA_2D2D_dfeast.cpp 

#ALL_OBJ=$(CPP_SRC:.cpp=.c_o)

EXE=$(CPP_SRC:.cpp=)

%.c_o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

SCSA_%: SCSA_%.c_o
	$(CC) $< -o $@ $(LIB_PATH) $(LIBS)

all: $(EXE)

$(EXE): $(ALL_OBJ)
        
clean:
	rm -f *.c_o $(EXE)
