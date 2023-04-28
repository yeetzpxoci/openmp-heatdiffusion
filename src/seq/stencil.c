#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>

#define REAL double

/* You may need a different method of timing if you are not on Linux. */
#define TIME(duration, fncalls)                                        \
    do {                                                               \
        struct timeval tv1, tv2;                                       \
        gettimeofday(&tv1, NULL);                                      \
        fncalls                                                        \
        gettimeofday(&tv2, NULL);                                      \
        duration = (REAL) (tv2.tv_usec - tv1.tv_usec) / 1000000 +    \
         (REAL) (tv2.tv_sec - tv1.tv_sec);                           \
    } while (0)

const REAL a = 0.1;
const REAL b = 0.2;
const REAL c = 0.3;

void Stencil(REAL **in, REAL **out, size_t n, int iterations)
{
    (*out)[0] = (*in)[0];
    (*out)[n - 1] = (*in)[n - 1];

    for (int t = 1; t <= iterations; t++) {
        /* Update only the inner values. */
        for (int i = 1; i < n - 1; i++) {
            (*out)[i] = a * (*in)[i - 1] +
                        b * (*in)[i] +
                        c * (*in)[i + 1];
        }

        /* The output of this iteration is the input of the next iteration (if there is one). */
        if (t != iterations) {
            REAL *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    REAL *in = calloc(n, sizeof(REAL));
    in[0] = 100;
    in[n - 1] = 1000;
    REAL *out = malloc(n * sizeof(REAL));

    double duration;
    TIME(duration, Stencil(&in, &out, n, iterations););
    printf("This took %lfs, or ??? Gflops/s\n", duration);

    free(in);
    free(out);

    return EXIT_SUCCESS;
}
