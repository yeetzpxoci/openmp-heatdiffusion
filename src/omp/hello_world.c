#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <omp.h>

int main(int argc, char **argv)
{
    char host[80];

    if (gethostname(host, 80) != 0) {
        host[0] = '\0';
    }

    #pragma omp parallel
    {
        printf("Hello world from %s, thread %d/%d running on CPU %d!\n",
                host, omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    }

    return EXIT_SUCCESS;
}
