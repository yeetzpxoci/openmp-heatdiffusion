#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    char host[80];

    if (gethostname(host, 80) != 0) {
        host[0] = '\0';
    }

    int s, p;

    MPI_Comm_rank(MPI_COMM_WORLD, &s);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    printf("Hello world from %s, rank %d/%d running on CPU %d!\n",
        host, s, p, sched_getcpu());

    MPI_Finalize();
}
