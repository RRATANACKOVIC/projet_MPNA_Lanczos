ROOT  = ./
BIN   = $(ROOT)bin/
SRC   = $(ROOT)src/
TEST  = $(ROOT)tests/
INC   = $(ROOT)inc/
CC    = gcc

#flags

FLIBS = -lm
#compilation

#programs

lanczos : main.o
	$(CC) -o $(BIN)lanczos $(BIN)main.o $(FLIBS)
main.o : $(SRC)main.c
	$(CC) -o $(BIN)main.o -c $(SRC)main.c $(FLIBS)


#Commands

clean :
	- rm -f bin/*
