
CC    = mpicc

#flags

FLIBS = -lm -lblas -llapacke
#compilation

#programs

lanczos : computations.o func.o main.o
	$(CC) -o lanczos computations.o func.o main.o $(FLIBS)
func.o : func.c
	$(CC) -o func.o -c func.c $(FLIBS)
computations.o : func.h computations.c
	$(CC) -o computations.o -c computations.c $(FLIBS)
main.o : func.h computations.h main.c
	$(CC) -o main.o -c main.c $(FLIBS)


#Commands

clean :
	- rm -f lanczos *.o
