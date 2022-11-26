ROOT  = ./
BIN   = $(ROOT)bin/
SRC   = $(ROOT)src/
TEST  = $(ROOT)tests/
INC   = $(ROOT)inc/
CC    = gcc

#flags

FLIBS = -lm -lblas
#compilation

#programs

lanczos : func.o main.o
	$(CC) -o $(BIN)lanczos $(BIN)func.o $(BIN)main.o $(FLIBS)
func.o : $(SRC)func.c
	$(CC) -o $(BIN)func.o -c $(SRC)func.c $(FLIBS)
main.o : $(INC)func.h $(SRC)main.c
	$(CC) -o $(BIN)main.o -c $(SRC)main.c $(FLIBS)


#Commands

clean :
	- rm -f bin/*
