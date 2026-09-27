/*
 * device_capacity.h -- how many work-groups of a kernel a device holds at once.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Included after device_primitives.h and before an algorithm core. Every core
 * runs VB_CAPACITY_PROBE instead of hashing when it is launched with repeats
 * 0, which no hash launch is, and the host caps its launch grid at the answer
 * (src/device/device.c).
 *
 * WHY THE GRID MUST FIT
 * =====================
 *
 * A launch sweeps the corpus `repeats` times, inside each work-item, so that a
 * small corpus still makes a launch long enough to time. When the device holds
 * every work-item of the launch at once, a work-item comes back to a message
 * only after the whole device has read the whole corpus, and the working set
 * is the corpus. When the grid is larger, it runs in waves, each wave repeating
 * only its own share -- and once a wave's share fits in a cache, every repeat
 * after the first is served from that cache. On an RTX PRO 2000 that made a
 * 256 MiB corpus report several times what the card's memory can deliver, out
 * of L2, and a 16 MiB one, which reads from L2, run out of L1; the checksum
 * held throughout, because every repeat really was hashed.
 *
 * WHY MEASURE IT, AND HERE
 * ========================
 *
 * How many work-groups fit depends on the kernel -- its registers above all --
 * as well as on the device. CUDA can report it and OpenCL cannot, so it is
 * measured, the same way under both, and inside the kernel it is for: a branch
 * of the same program is compiled with that program's resources, which a
 * separate probe kernel would not be.
 *
 * The first work-item of each work-group counts itself in, then watches how
 * many work-groups are in and not yet out, until the whole launch is in or the
 * first one leaves. The host launches more work-groups than any device holds,
 * so the first wave fills the device and nothing leaves until `spin` reads
 * have passed; the most any work-group saw is the capacity. `started` is read
 * before `finished`, which can only undercount -- and an undercount gives a
 * smaller grid, which is still honest.
 *
 * The rest of the work-group waits at a barrier meanwhile, as the caller
 * writes it. Work-items that simply returned would hand back their registers
 * and scheduling slots, and more work-groups would fit than any hash launch
 * does: an RTX PRO 2000 so measured appeared to hold more work-items per SM
 * than its hardware can.
 *
 * count[0] is work-groups started, count[1] finished, count[2] the most seen
 * in at once. The host zeroes them first.
 */

#define VB_CAPACITY_PROBE(count, spin)                                   \
    do {                                                                 \
        volatile VB_GLOBAL vb_u32 *cp_ = (count);                        \
        const vb_u32 groups_ = (vb_u32) VB_GROUP_COUNT();                \
        vb_u32 most_ = 0;                                                \
        VB_ATOMIC_INC(&cp_[0]);                                          \
        for (vb_u32 i_ = 0; i_ < (spin); i_++) {                         \
            const vb_u32 started_ = cp_[0];                              \
            const vb_u32 finished_ = cp_[1];                             \
            if (started_ - finished_ > most_)                            \
                most_ = started_ - finished_;                            \
            if (started_ == groups_ || finished_ != 0)                   \
                break;                                                   \
        }                                                                \
        VB_ATOMIC_MAX(&cp_[2], most_);                                   \
        VB_ATOMIC_INC(&cp_[1]);                                          \
    } while (0)
