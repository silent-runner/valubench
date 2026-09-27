/*
 * fail_pthread_create.c -- make one pthread_create() fail, on purpose.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Preload this and set VB_FAIL_CREATE=N to make the Nth call fail with
 * EAGAIN, one-based. Every other call is forwarded. The preload is
 * LD_PRELOAD on Linux and DYLD_INSERT_LIBRARIES on macOS; the Makefile
 * chooses.
 *
 * The point is the *non-contiguous* failure. A test that exhausts the thread
 * limit gets a contiguous prefix of successes, which the old code happened to
 * handle; the defect only appears when thread 1 fails and thread 2 starts, and
 * nothing but an interposer produces that.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned calls;

/*
 * Whether this call is the one to refuse. A refusal is announced on stderr so
 * the check can tell a failure it injected from a loader that never ran this
 * library: macOS ignores LD_PRELOAD, and for as long as the Makefile used it
 * there this check passed while injecting nothing.
 */
static int refuse_this_call(void)
{
    const char *want = getenv("VB_FAIL_CREATE");
    unsigned n = ++calls;

    if (want && n == (unsigned) atoi(want)) {
        fprintf(stderr, "fail_pthread_create: refused call %u\n", n);
        return 1;
    }
    return 0;
}

/*
 * A watchdog, because what this provokes can be a hang: a pool waiting at a
 * barrier for a worker that never started. GNU timeout bounded the run from the
 * Makefile, but macOS has no timeout, and even a timeout from elsewhere is no
 * help if it is a system binary -- those shed DYLD_* variables, so whatever
 * they launch runs without this library. An alarm armed inside the process
 * needs neither. Its default action ends the process however its threads are
 * blocked, and the shell reports 128 + SIGALRM, which the Makefile treats as
 * a hang.
 */
__attribute__((constructor))
static void arm_watchdog(void)
{
    alarm(60);
}

#if defined(__APPLE__)

/*
 * A two-level namespace binds each import to the library that defined it at
 * link time, so a pthread_create defined here would replace nothing. dyld's
 * interpose section is the mechanism for this instead: every other image's
 * references to the second pointer are rebound to the first, and this image's
 * are not, so the wrapper reaches the real function by its own name.
 */
static int interposed_pthread_create(pthread_t *t, const pthread_attr_t *a,
                                     void *(*fn)(void *), void *arg)
{
    if (refuse_this_call())
        return EAGAIN;

    return pthread_create(t, a, fn, arg);
}

typedef int (*create_fn)(pthread_t *, const pthread_attr_t *,
                         void *(*)(void *), void *);

__attribute__((used, section("__DATA,__interpose")))
static const struct { create_fn replacement, replacee; } interposers[] = {
    { interposed_pthread_create, pthread_create },
};

#else

int pthread_create(pthread_t *t, const pthread_attr_t *a,
                   void *(*fn)(void *), void *arg)
{
    typedef int (*create_fn)(pthread_t *, const pthread_attr_t *,
                             void *(*)(void *), void *);
    static create_fn real;
    if (!real) {
        /* ISO C forbids assigning void* to a function pointer directly; the
           copy is the portable spelling POSIX itself recommends. */
        void *sym = dlsym(RTLD_NEXT, "pthread_create");
        memcpy(&real, &sym, sizeof real);
    }

    if (refuse_this_call())
        return EAGAIN;

    return real(t, a, fn, arg);
}

#endif
