/*
 *      ANSI C implementation of vector operations.
 *
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id$ */

#include <stdlib.h>
#include <memory.h>
#include "cmfrec.h"

// #if     LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
// #ifdef USE_FLOAT
// #define fsigndiff(x, y) (((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
// #else
#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
// #endif/*LBFGS_IEEE_FLOAT*/
// #endif

// inline static void* vecalloc(size_t size)
// {
//     void *memblock = malloc(size);
//     if (memblock) {
//         memset(memblock, 0, size);
//     }
//     return memblock;
// }
#define vecalloc calloc

inline static void vecfree(void *memblock)
{
    free(memblock);
}

inline static void vecset(real_t *x, const real_t c, const size_t n)
{
    size_t i;
    for (i = 0;i < n;++i) {
        x[i] = c;
    }
}

inline static void veccpy(real_t *y, const real_t *x, const size_t n)
{
    memcpy(y, x, n*sizeof(real_t));
}

inline static void vecncpy(real_t *restrict y, const real_t *restrict x, const size_t n)
{
    size_t i;
    for (i = 0;i < n;++i) {
        y[i] = -x[i];
    }
}

inline static void vecadd(real_t *y, const real_t *x, const real_t c, const size_t n)
{
    if (n < (size_t)INT_MAX)
        cblas_taxpy((int)n, c, x, 1, y, 1);
    else
        for (size_t ix = 0; ix < n; ix++)
            y[ix] += c*x[ix];
}

inline static void vecdiff(real_t *restrict z, const real_t *restrict x, const real_t *restrict y, const size_t n)
{
    size_t i;
    for (i = 0;i < n;++i) {
        z[i] = x[i] - y[i];
    }
}

inline static void vecscale(real_t *y, const real_t c, const size_t n)
{
    if (n < (size_t)INT_MAX)
        cblas_tscal((int)n, c, y, 1);
    else
        for (size_t ix = 0; ix < n; ix++)
            y[ix] *= c;
}

inline static void vecmul(real_t *restrict y, const real_t *restrict x, const size_t n)
{
    size_t i;
    for (i = 0;i < n;++i) {
        y[i] *= x[i];
    }
}

inline static void vecdot(real_t* s, const real_t *x, const real_t *y, const size_t n)
{
    if (n < (size_t)INT_MAX)
        *s = cblas_tdot((int)n, x, 1, y, 1);
    else {
        long double res = 0;
        for (size_t ix = 0; ix < n; ix++)
            res += x[ix]*y[ix];
        *s = (real_t)res;
    }
}

inline static void vec2norm(real_t* s, const real_t *x, const size_t n)
{
    if (n < (size_t)INT_MAX)
        *s = cblas_tnrm2(n, x, 1);
    else {
        long double res = 0;
        for (size_t ix = 0; ix < n; ix++)
            res += square(x[ix]);
        *s = (real_t)sqrtl(res);
    }
}

inline static void vec2norminv(real_t* s, const real_t *x, const size_t n)
{
    vec2norm(s, x, n);
    *s = (real_t)(1.0 / *s);
}
