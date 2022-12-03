ROOT  = ./
BIN   = $(ROOT)bin/
SRC   = $(ROOT)src/
TEST  = $(ROOT)tests/
INC   = $(ROOT)inc/
CC    = gcc

#flags

FLIBS = -lm -lblas -llapacke
#compilation

#programs

lanczos : computations.o func.o main.o
	$(CC) -o $(BIN)lanczos $(BIN)computations.o $(BIN)func.o $(BIN)main.o $(FLIBS)
func.o : $(SRC)func.c
	$(CC) -o $(BIN)func.o -c $(SRC)func.c $(FLIBS)
computations.o : $(INC)func.h $(SRC)computations.c
	$(CC) -o $(BIN)computations.o -c $(SRC)computations.c $(FLIBS)
main.o : $(INC)func.h $(INC)computations.h $(SRC)main.c
	$(CC) -o $(BIN)main.o -c $(SRC)main.c $(FLIBS)


#Commands

clean :
	- rm -f bin/*
