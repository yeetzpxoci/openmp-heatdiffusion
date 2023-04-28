#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>

int main(int argc, char **argv)
{
    char host[80];

    if (gethostname(host, 80) != 0) {
        host[0] = '\0';
    }

    printf("Hello world from %s, CPU %d!\n", host, sched_getcpu());

    return EXIT_SUCCESS;
}
