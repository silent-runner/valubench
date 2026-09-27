/*
 * vb_threads.h -- the threading facilities this harness uses that POSIX does
 * not portably provide, behind one interface so bench.c never spells a
 * platform primitive itself.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Three things here are not portable, and each is isolated so Linux makes
 * exactly the calls it made before and every other platform gets a defined
 * fallback instead of a build break. "The same calls" is not "the same object
 * code": pin_self now inlines into its callers, so bench.o differs. What
 * design.md finding 10 holds to bit-identity is the hash loop, run_slice,
 * and nothing in this file is called from it.
 *
 *   - Barriers are an *optional* POSIX feature (_POSIX_BARRIERS). glibc has
 *     them; several libcs do not. The fallback is a mutex and condition
 *     variable doing the same job, with a generation counter so the barrier
 *     is reusable -- without it a fast worker reaching the next wait before a
 *     slow one has left the previous one would sail straight through, and in
 *     this harness that means a worker starting its slice before the driver
 *     has set `reps`: a plausible wrong number, the one thing this benchmark
 *     must never produce.
 *
 *     Both waits are inside the timed interval. pool_run reads the clock
 *     before releasing the workers and again after collecting them, so every
 *     sample includes two barrier crossings. That cost is paid once per
 *     sample, not once per hash, which keeps it small beside a sample sized
 *     to --time-ms -- but it is not zero, and the fallback pays more of it
 *     than a futex barrier does, because each woken waiter must retake the
 *     mutex on its way out.
 *
 *   - Thread pinning is not standard at all. Linux has the GNU extension
 *     pthread_setaffinity_np; there is no portable equivalent. vb_thread_pin
 *     keeps the caller's contract (0 pinned, -1 refused) where an API exists,
 *     and where none does it reports unsupported through
 *     vb_thread_pin_supported() rather than faking a refusal -- a machine with
 *     no affinity API did not refuse a request, it was never asked, and the
 *     two are different facts about a result.
 *
 *   - Reading the calling thread's own CPU mask is Linux-specific
 *     (sched_getaffinity). vb_self_cpu_mask returns -1 where the notion does
 *     not exist, and the caller falls back to the online set exactly as it
 *     already does when the read fails.
 */

#ifndef VALUBENCH_VB_THREADS_H
#define VALUBENCH_VB_THREADS_H

#include <pthread.h>

#if defined(__linux__)

#include <sched.h>

typedef pthread_barrier_t vb_barrier;

static inline int vb_barrier_init(vb_barrier *b, unsigned count)
{
    return pthread_barrier_init(b, NULL, count);
}

static inline int vb_barrier_wait(vb_barrier *b)
{
    return pthread_barrier_wait(b);
}

static inline int vb_barrier_destroy(vb_barrier *b)
{
    return pthread_barrier_destroy(b);
}

/* 0 pinned, -1 the CPU was refused. Identical to the affinity call bench.c
   used inline; a refusal must stay visible, since a run that says it pinned
   and did not is a different measurement. */
static inline int vb_thread_pin(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((unsigned) cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0
           ? 0 : -1;
}

static inline int vb_thread_pin_supported(void) { return 1; }

/* Enumerate the calling thread's allowed CPUs in ascending order. Returns the
   count written, or -1 if the mask could not be read. */
static inline int vb_self_cpu_mask(int *out, unsigned max)
{
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) != 0)
        return -1;

    unsigned n = 0;
    for (unsigned cpu = 0; cpu < CPU_SETSIZE && n < max; cpu++)
        if (CPU_ISSET(cpu, &set))
            out[n++] = (int) cpu;
    return (int) n;
}

#else  /* no POSIX barriers, no affinity API: macOS and anything else */

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    unsigned        threshold;
    unsigned        waiting;
    unsigned        generation;
} vb_barrier;

static inline int vb_barrier_init(vb_barrier *b, unsigned count)
{
    int rc = pthread_mutex_init(&b->mutex, NULL);
    if (rc != 0)
        return rc;
    rc = pthread_cond_init(&b->cond, NULL);
    if (rc != 0) {
        pthread_mutex_destroy(&b->mutex);
        return rc;
    }
    b->threshold  = count;
    b->waiting    = 0;
    b->generation = 0;
    return 0;
}

static inline int vb_barrier_wait(vb_barrier *b)
{
    pthread_mutex_lock(&b->mutex);
    unsigned gen = b->generation;

    if (++b->waiting == b->threshold) {
        b->waiting = 0;
        b->generation++;
        pthread_cond_broadcast(&b->cond);
    } else {
        /* A condition variable may wake spuriously; the generation is the
           predicate that says the barrier genuinely opened. */
        while (gen == b->generation)
            pthread_cond_wait(&b->cond, &b->mutex);
    }

    pthread_mutex_unlock(&b->mutex);
    return 0;
}

static inline int vb_barrier_destroy(vb_barrier *b)
{
    pthread_cond_destroy(&b->cond);
    pthread_mutex_destroy(&b->mutex);
    return 0;
}

/* No affinity API. THREAD_AFFINITY_POLICY on Darwin is advisory and Apple
   silicon ignores it, so there is nothing to attempt and nothing to fail;
   vb_thread_pin_supported() carries that fact to the result instead. */
static inline int vb_thread_pin(int cpu) { (void) cpu; return 0; }

static inline int vb_thread_pin_supported(void) { return 0; }

static inline int vb_self_cpu_mask(int *out, unsigned max)
{
    (void) out;
    (void) max;
    return -1;
}

#endif

#endif /* VALUBENCH_VB_THREADS_H */
