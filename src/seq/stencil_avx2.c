#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

//#define CHECK

/* You may need a different method of timing if you are not on Linux. */
#define TIME(duration, fncalls)                                        \
    do {                                                               \
        struct timeval tv1, tv2;                                       \
        gettimeofday(&tv1, NULL);                                      \
        fncalls                                                        \
        gettimeofday(&tv2, NULL);                                      \
        duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +    \
         (double) (tv2.tv_sec - tv1.tv_sec);                           \
    } while (0)

const double a = 0.3;
const double b = 0.5;
const double c = 0.4;

/* We split up the stencil in smaller stencils, of roughly SPACEBLOCK size,
 * and treat them for TIMEBLOCK iterations. Play around with these. Do the considerations
 * change when parallelising? */
const int SPACEBLOCK = 1250;
const int TIMEBLOCK = 100;

/* Takes buffers *in, *out of size n + iterations.
 * out[0: n - 1] is the first part of the stencil of in[0, n + iterations - 1]. */
void Left(double **in, double **out, size_t n, int iterations)
{
    (*out)[0] = (*in)[0];

    for (int t = 1; t <= iterations; t++) {
        for (size_t i = 1; i < n + iterations - t; i++) {
            (*out)[i] = a * (*in)[i - 1] + b * (*in)[i] + c * (*in)[i + 1];
        }

        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

/* This version is written with avx2 in mind. Each vector register can fit 4 doubles.
 * So we need 3 loads (in[i - 1], in[i], in[i + 1]) and one load (out[i]) to calculate
 * out[i] for i, i + 1, i + 2, i + 3. This takes 3 instructions.
 *
 * Instead, we can calculate 3 iterations at a time. That means out[i] is a linear combination
 * of in[i - 3], in[i - 2], in[i - 1], in[i], in[i + 1], in[i + 2], in[i + 3].
 * This fits in the 16 registers as we need 7 loads of in, 7 constants, and one register
 * for the result. The nice thing is that in[i + 1] = in[i + 4 - 3], in[i + 2] = in[i + 4 - 2],
 * in[i + 3] = in[i + 4 - 1]. So we can keep these in the registers for the next iteration.
 * So really, we only have 4 loads, and one store on 7 instructions. The compiler cannot do this,
 * so we use intel intrinics. You can calculate the constants by repeated subsitution, or by
 * multiplying three matrices:
 * https://www.wolframalpha.com/input?i=matrix+multiplication+calculator&assumption=%7B%22F%22%2C+%22MatricesOperations%22%2C+%22theMatrix3%22%7D+-%3E%22%7B%7Ba%2C+b%2C+c%2C+0%2C+0%2C+0%2C+0%7D%2C+%7B0%2C+a%2C+b%2C+c%2C+0%2C+0%2C+0%7D%2C+%7B0%2C+0%2C+a%2C+b%2C+c%2C+0%2C+0%7D%2C+%7B0%2C+0%2C+0%2C+a%2C+b%2C+c%2C+0%7D%2C%7B0%2C+0%2C+0%2C+0%2C+a%2C+b%2C+c%7D%7D%22&assumption=%7B%22F%22%2C+%22MatricesOperations%22%2C+%22theMatrix2%22%7D+-%3E%22%7B%7Ba%2Cb%2Cc%2C0%2C0%7D%2C%7B0%2Ca%2Cb%2Cc%2C0%7D%2C%7B0%2C0%2Ca%2Cb%2Cc%7D%7D%22&assumption=%7B%22F%22%2C+%22MatricesOperations%22%2C+%22theMatrix1%22%7D+-%3E%22%7B%7Ba%2C+b%2C+c%7D%7D%22*/
void Middle(double **in, double **out, size_t n, int iterations)
{
    const double c0 = a * a * a;
    const double c1 = 3 * a * a * b;
    const double c2 = a * a * c + a * (2 * a * c + b * b) + 2 * a * b * b;
    const double c3 = b * (2 * a * c + b * b) + 4 * a * b * c;
    const double c4 = c * (2 * a * c + b * b) + a * c * c + 2 * b * b * c;
    const double c5 = 3 * b * c * c;
    const double c6 = c * c * c;

    __m256d ymm0 = _mm256_broadcast_sd(&c0);
    __m256d ymm1 = _mm256_broadcast_sd(&c1);
    __m256d ymm2 = _mm256_broadcast_sd(&c2);
    __m256d ymm3 = _mm256_broadcast_sd(&c3);
    __m256d ymm4 = _mm256_broadcast_sd(&c4);
    __m256d ymm5 = _mm256_broadcast_sd(&c5);
    __m256d ymm6 = _mm256_broadcast_sd(&c6);
    __m256d ymm7;
    __m256d ymm8;
    __m256d ymm9;
    __m256d ymm10;
    __m256d ymm11;
    __m256d ymm12;
    __m256d ymm13;

    /* We do a few iterations the slow way first */
    /* So we can do the rest in steps of 3 */
    int pre_iters = iterations % 3;
    /* So we can load in[i - 3]. */
    pre_iters = (pre_iters < 3) ? 3 + pre_iters : pre_iters;
    int t = 1;
    for (; t <= pre_iters; t++) {
        for (size_t i = t; i < n + 2 * iterations - t; i++) {
            (*out)[i] = a * (*in)[i - 1] +
                        b * (*in)[i] +
                        c * (*in)[i + 1];
        }
        double *temp = *in;
        *in = *out;
        *out = temp;
    }

    /* Steps 1, 2, need to be done */
    for (; t < iterations; t += 3) {
        /* Preload */
        ymm7 = _mm256_loadu_pd(*in + t - 3);
        ymm8 = _mm256_loadu_pd(*in + t - 2);
        ymm9 = _mm256_loadu_pd(*in + t - 1);

        /* In one of these loops, we calculate out[j] for j = i, ..., i + 7. We assume
         * ymm7, ymm8, ymm9 are already loaded with in[i - 3], in[i - 2], in[i - 1]. */
        size_t i = t;
        for (; i < n + 2 * iterations - t - 8; i += 8) {
            /* Calculation of out[j] for j = i, ..., i + 3. */
            ymm10 = _mm256_loadu_pd(*in + i);
            ymm11 = _mm256_loadu_pd(*in + i + 1);
            ymm12 = _mm256_loadu_pd(*in + i + 2);
            ymm13 = _mm256_loadu_pd(*in + i + 3);

            ymm10 = _mm256_mul_pd(ymm10, ymm3); // ymm10 will hold (*out)[i]
            ymm10 = _mm256_fmadd_pd(ymm7, ymm0, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm8, ymm1, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm9, ymm2, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm11, ymm4, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm12, ymm5, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm13, ymm6, ymm10);

            _mm256_storeu_pd(*out + i, ymm10);

            /* Calculation of out[j] for j = i + 4, ..., i + 7. */
            ymm10 = _mm256_loadu_pd(*in + i + 4);
            ymm7 = _mm256_loadu_pd(*in + i + 5);
            ymm8 = _mm256_loadu_pd(*in + i + 6);
            ymm9 = _mm256_loadu_pd(*in + i + 7);

            ymm10 = _mm256_mul_pd(ymm10, ymm3); // ymm10 will hold (*out)[i + 4]
            ymm10 = _mm256_fmadd_pd(ymm11, ymm0, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm12, ymm1, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm13, ymm2, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm7, ymm4, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm8, ymm5, ymm10);
            ymm10 = _mm256_fmadd_pd(ymm9, ymm6, ymm10);

            _mm256_storeu_pd(*out + (i + 4), ymm10);
        }

        /* Remaining part if n + 2 * iterations - 2 * t is not divisible by 8. */
        for (; i < n + 2 * iterations - t; i++) {
            (*out)[i] = c0 * (*in)[i - 3] +
                        c1 * (*in)[i - 2] +
                        c2 * (*in)[i - 1] +
                        c3 * (*in)[i] +
                        c4 * (*in)[i + 1] +
                        c5 * (*in)[i + 2] +
                        c6 * (*in)[i + 3];
        }

        double *temp = *in;
        *in = *out;
        *out = temp;
    }

    double *temp = *in;
    *in = *out;
    *out = temp;
}

/* Takes buffers *in, *out of size n + iterations.
 * out[iterations: n + iterations - 1] is the last part of
 * the stencil of in[0, n + iterations - 1]. */
void Right(double **in, double **out, size_t n, int iterations)
{
    (*out)[n + iterations - 1] = (*in)[n + iterations - 1];

    for (int t = 1; t <= iterations; t++) {
        for (size_t i = t; i < n + iterations - 1; i++) {
            (*out)[i] = a * (*in)[i - 1] + b * (*in)[i] + c * (*in)[i + 1];
        }

        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

void StencilBlocked(double **in, double **out, size_t n, int iterations)
{
    double *inBuffer = malloc((SPACEBLOCK + 2 * iterations) * sizeof(double));
    double *outBuffer = malloc((SPACEBLOCK + 2 * iterations) * sizeof(double));

    for (size_t block = 0; block < n / SPACEBLOCK; block++) {
        if (block == 0) {
            memcpy(inBuffer, *in, (SPACEBLOCK + iterations) * sizeof(double));
            Left(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out, outBuffer, SPACEBLOCK * sizeof(double));
        } else if (block == n / SPACEBLOCK - 1) {
            memcpy(inBuffer, *in + block * SPACEBLOCK - iterations,
                    (SPACEBLOCK + iterations) * sizeof(double));
            Right(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out + block * SPACEBLOCK, outBuffer + iterations, SPACEBLOCK * sizeof(double));
        } else {
            memcpy(inBuffer, *in + block * SPACEBLOCK - iterations,
                    (SPACEBLOCK + 2 * iterations) * sizeof(double));
            Middle(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out + block * SPACEBLOCK, outBuffer + iterations, SPACEBLOCK * sizeof(double));
        }
    }

    free(inBuffer);
    free(outBuffer);
}

void Stencil(double **in, double **out, size_t n, int iterations)
{
    int rest_iters = iterations % TIMEBLOCK;
    if (rest_iters != 0) {
        StencilBlocked(in, out, n, rest_iters);
        double *temp = *out;
        *out = *in;
        *in = temp;
    }

    for (int t = rest_iters; t < iterations; t += TIMEBLOCK) {
        StencilBlocked(in, out, n, TIMEBLOCK);
        double *temp = *out;
        *out = *in;
        *in = temp;
    }

    double *temp = *out;
    *out = *in;
    *in = temp;
}

void StencilSlow(double **in, double **out, size_t n, int iterations)
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
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

#ifdef CHECK
bool equal(double *x, double *y, size_t n, double error)
{
    for (size_t i = 0; i < n; i++) {
        if (fabs(x[i] - y[i]) > error) {
            printf("Index %zu: %lf != %lf\n", i, x[i], y[i]);
            return false;
        }
    }

    return true;
}
#endif

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    if (n % SPACEBLOCK != 0) {
        printf("I am lazy, so assumed that SPACEBLOCK divides n. Suggestion: n = %ld\n",
                n / SPACEBLOCK * SPACEBLOCK);
        return EXIT_FAILURE;
    }

    double *in = calloc(n, sizeof(double));
    in[0] = 100;
    in[n / 2] = n;
    in[n - 1] = 1000;
    double *out = malloc(n * sizeof(double));

    double duration;
    TIME(duration, Stencil(&in, &out, n, iterations););
    printf("Faster version took %lfs, or ??? Gflops/s\n", duration);

#ifdef CHECK
    double *in2 = calloc(n, sizeof(double));
    in2[0] = 100;
    in2[n / 2] = n;
    in2[n - 1] = 1000;
    double *out2 = malloc(n * sizeof(double));
    TIME(duration, StencilSlow(&in2, &out2, n, iterations););
    printf("Slow version took %lfs, or ??? Gflops/s\n", duration);
    printf("Checking whether they computed the same result with error 0.0001...\n");
    if (equal(out, out2, n, 0.0001)) {
        printf("They are (roughly) equal\n");
    }
    free(in2);
    free(out2);
#endif

    free(in);
    free(out);

    return EXIT_SUCCESS;
}
