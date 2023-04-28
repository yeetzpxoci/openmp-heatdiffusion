CC = gcc
MPICC = mpicc
FLAGS = -Ofast -march=native -mtune=native -Wall -Werror
LFLAGS = -lm

SEQ_SRC = $(wildcard src/seq/*.c)
SEQ = $(patsubst src/seq/%.c, bin/%_seq, $(SEQ_SRC))
OMP_SRC = $(wildcard src/omp/*.c)
OMP = $(patsubst src/omp/%.c, bin/%_omp, $(OMP_SRC))
MPI_SRC = $(wildcard src/mpi/*.c)
MPI = $(patsubst src/mpi/%.c, bin/%_mpi, $(MPI_SRC))

.PHONY: all seq omp mpi clean

all: seq omp mpi

seq: $(SEQ)

omp: $(OMP)

mpi: $(MPI)

bin/%_seq: src/seq/%.c
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS)

bin/%_omp: src/omp/%.c
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS) -fopenmp

bin/%_mpi: src/mpi/%.c
	$(MPICC) $(FLAGS) $^ -o $@ $(LFLAGS)

clean:
	$(RM) bin/* mpi_hello_* *.out
