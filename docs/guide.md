# valubench user guide

Long-form usage. The [README](../README.md) covers what valubench measures and
how to run it; this covers everything else the tool can do.

## The three axes

| Flag | Moves | Effect |
|---|---|---|
| `--iterations N` | compute per hash | more work, no extra memory touched |
| `--message-bytes L` | compute **and** bytes per hash | more blocks, more data read |
| `--working-set-kb K` | memory footprint | how many messages, so which cache level holds them |

Together they trace the surface that locates the compute-bound / memory-bound
crossover, which is the second goal in [docs/design.md](design.md).

### Message length

Block-count boundaries dominate: crossing 55→56 bytes doubles the blocks and
roughly halves hashes/sec, while compressions/sec — the real invariant — stays in
a band (measured).

Two-block messages are the worst case: the second block is nearly all padding,
so half the compressions do almost no useful byte-work.

### Working set

Sweeping `--working-set-kb` at fixed message length walks the corpus through the
cache hierarchy (measured).

**Whether MD5 becomes memory-bound depends on the kernel, not on MD5.** One
compression is several hundred integer operations per 64 bytes, which puts the
workload far to the right of the roofline ridge point *at the rate the machine
can hash* — so a slow kernel never reaches the ridge and a fast one does. On
the development N100 and on Graviton3 the corpus leaving cache costs a tenth of
throughput and almost nothing respectively; on a Zen 5 core running AVX-512 it
costs 44% single-threaded, and across all sixteen cores of a desktop Zen 5 part
it costs a factor of eight, flattening onto a DRAM floor. On that same machine,
in the same sweep, AVX2 and scalar barely move.

This section previously said MD5 "barely becomes memory-bound" and that a
genuine memory-bound regime would need much larger messages or a
bandwidth-hungry companion kernel. That was a generalisation from the two
slowest parts measured, and it is wrong on a wide one — see
[design.md](design.md) §2a, which works through what it means for goal 2.
Widening the datapath moves you along the roofline toward the ridge, which is
the same thing as saying a faster kernel is easier to starve.

**The default is not a neutral choice on every part.** `--working-set-kb 1024`
resolves to a 1,008 KiB corpus, which on a core with 1 MiB of L2 sits on the
capacity boundary rather than clear of either side. What that costs depends on
the kernel: on a desktop Zen 5 part `md5/avx512-s4` loses about 4% there while
`s6` loses 31% and `s8` 40%, and AVX2 and scalar do not notice. Two
consequences worth knowing before quoting a single-thread number:

- **The stream ordering inverts across the boundary.** In L2 the ranking is
  s8 > s6 > s4; past it the ranking reverses and s4 is fastest. So
  **autotune's choice of kernel depends on `--working-set-kb`**, and the kernel
  it picks at the default is not the one that wins on a corpus that fits.
- **A single-thread measurement should name a corpus that clearly fits or
  clearly does not.** Sitting on the boundary also costs reproducibility —
  see "What the coefficient of variation does not cover" below.

Multi-threaded runs are affected far less, because the corpus is split across
workers: 1,008 KiB over 32 threads is about 31 KiB each, comfortably inside L1.

**Very long messages need a bigger corpus, not just a bigger message.** A
kernel processes `lanes x streams` messages per group, and a thread with fewer
messages than that leaves lanes idle. The whole pool therefore wants at least
`threads x lanes x streams` messages before any lane-level figure means
anything — 3,072 on a 32-thread part running a 16-lane, 6-stream kernel. At
`--message-bytes 16375` the default corpus holds 768, so the point reports
roughly a third of the compression rate every shorter message length reaches,
and the deficit is starvation rather than anything about long messages.
Growing the corpus recovers it monotonically. This is the CPU form of the
device-side limit described under "Saturating the device" below.

**Very short messages cap the corpus from the other side.** A message carries
its index in its first four bytes, so a message of one to three bytes has only
256, 65,536 or 16,777,216 distinct values. A larger corpus would repeat
messages, and repeated digests cancel in the XOR checksum -- 1,536 one-byte
messages are 256 values six times over and fingerprint to all zeros -- so the
tool refuses it. One-byte messages cannot be verified at any working set, since
the smallest corpus is 768 messages; two-byte messages allow up to 4 MiB.

## Tuning compute intensity

`--iterations N` chains N MD5s per hash, feeding each digest back as the first 16
bytes **of the same message** rather than hashing the bare digest. That
distinction is what keeps the knob linear: hashing a 16-byte digest would
collapse to one block with 12 of 16 words constant, so later iterations would
cost less than the first. Feeding back into the full message keeps the
instruction mix and block count identical every time. It also means
`--iterations > 1` requires `--message-bytes >= 16`.

Compressions/sec stays flat while hashes/sec falls proportionally, which is what
a linear compute knob should do
(measured).

This is the compute axis; message length and working set are the others.

## Sweeping

[`tools/sweep.py`](../tools/sweep.py) walks a grid over the three axes, collects the
JSON, and writes one CSV row per point — the shape mixbench uses, so it feeds
straight into whatever plots it. Standard library only; Python never enters a
timed region, and the C binary stays fully usable without it.

```
./tools/sweep.py --kernel avx2-s4 --threads 1     --message-bytes 64:4096:*2 --iterations 1,4,16 --csv surface.csv
```

Axis syntax is a list, a geometric range, or an arithmetic one:

| Spec | Expands to |
|---|---|
| `55` | 55 |
| `55,1015,4087` | those three |
| `64:4096:*2` | 64, 128, 256, … 4096 |
| `1:10:+3` | 1, 4, 7, 10 |

CSV goes to stdout unless `--csv` names a file, in which case stdout gets the
table instead. Progress always goes to stderr, so redirecting is safe.
`--dry-run` shows the grid and a time estimate first; `--kernel` skips autotune
per point and is much faster for large grids.

The driver keeps points that exit noisy (marking them in a `status` column
rather than discarding a real measurement), skips combinations the workload
forbids, and **aborts on a verification failure** — if the hardware computed a
wrong digest once, every later point is suspect too. `--keep-going` overrides.

A sweep across a 74x range of message sizes and 8x iterations, single core:

```
  msg B    blk   iter   thr     WS KiB       kernel         MH/s        MC/s
-------  -----  -----  ----  ---------  -----------  -----------  ----------
     55      1      1     1       1020      avx2-s4        42.38       42.38
     55      1      8     1       1020      avx2-s4         5.35       42.80
    247      4      1     1       1008      avx2-s4        10.42       41.68
   1015     16      1     1        960      avx2-s4         2.75       43.98
   4087     64      8     1        768      avx2-s4         0.09       44.53
```

Hashes/sec spans nearly three orders of magnitude while compressions/sec stays
inside a narrow band — which is the invariant both axes were designed to
preserve.

Everything the driver needs to know about the binary, it asks the binary for.
`valubench --list --json` reports the algorithms and their geometry, every
kernel and whether it runs here, the parameter limits, the defaults and the exit
codes — so a caller never has to carry a transcribed copy of those facts, which
is how the driver used to get the per-algorithm message-length minimum wrong.

## Comparing runs

One run prints one number; the question that matters is usually whether it
moved. [`tools/compare.py`](../tools/compare.py) pairs points from two result sets —
single JSON files, directories, or sweep CSVs — and reports the change:

```
$ ./tools/compare.py before.json after/
point                            base MH/s      new MH/s     delta    noise  verdict
---------------------------------------------------------------------------------
md5-full-55x1 md5/avx2-s3 1t         ...           ...      -30.0%     1.4%  SLOWER
```

Two things keep it honest. Points are paired by **workload id**, so runs that
measured different work are never compared, and a **checksum mismatch is a hard
failure** rather than a delta — the XOR checksum is invariant across lanes,
streams, threads and devices, so two runs of the same workload must agree on any
machine. A delta is only called a change when it clears both the 10% bar
[docs/design.md](design.md) sets and the run-to-run noise the two runs reported; on a
loaded machine that second bar frequently disqualifies the first.

Exit status is 0 when nothing regressed, 1 on a regression, 3 when the two sets
are not comparable — so it drops into a script.

## GPUs

OpenCL is `dlopen`'d at runtime — never linked — so the build needs no SDK and
the same binary runs on a machine with no GPU, no driver, or no OpenCL at all.
Build dependencies stay at a C compiler, libdl and libpthread.

If `<CL/cl.h>` is installed (`opencl-headers`), the build uses it; otherwise it
falls back to declarations in [include/vb_cl.h](../include/vb_cl.h). Runtime
behaviour is identical either way — `make config` reports which was used, and
both produce the same checksum. Linking `-lOpenCL` is deliberately avoided: it
would make the binary fail to *start* without libOpenCL, which is the opposite
of "GPU kernels when the system has OpenCL, CPU-only when it does not".

```
$ ./build/valubench --list-devices
[0] Intel(R) UHD Graphics
     vendor    Intel(R) Corporation
     type      GPU, 24 compute units @ 750 MHz
     memory    14373 MiB global, 4095 MiB max allocation
     platform  Intel(R) OpenCL Graphics (OpenCL 3.0 )
     driver    23.43.027642, device OpenCL 3.0 NEO
```

Device kernels appear in `--list` as `ocl-s1..s4` and take part in autotune
alongside the CPU ones.

**Multiple devices run concurrently.** `--device` takes an index, a
comma-separated list, or `all` (the default). The corpus is split into
contiguous group ranges, one per device — exactly as the CPU thread pool splits
it across cores — and each device verifies its own slice against its own
reference value. Because XOR is associative, folding the slices reproduces the
single-device checksum, so **the fingerprint is identical at any device count**:

```
$ valubench --json --kernel ocl-s1 --device 0    | grep checksum
    "checksum": "955e84cbbc05470019604a2bd9ff2821"
$ valubench --json --kernel ocl-s1 --device 0,0  | grep checksum
    "checksum": "955e84cbbc05470019604a2bd9ff2821"
```

Devices are dispatched with an enqueue-all-then-collect-all pass rather than a
loop of blocking runs, which would serialise them.

The split is equal, not weighted by device speed, so a heterogeneous set is
paced by its slowest member. Recorded as a known limitation rather than averaged
away silently.

**The checksum is identical on CPU and GPU**, which is what makes the whole
verification design pay off — the existing reference validates device kernels
with no extra machinery:

```
scalar-s1  955e84cbbc05470019604a2bd9ff2821
avx2-s4    955e84cbbc05470019604a2bd9ff2821
ocl-s1     955e84cbbc05470019604a2bd9ff2821
ocl-s3     955e84cbbc05470019604a2bd9ff2821
```

The corpus is uploaded once outside the timed region, and digests are reduced
to one `uint4` per work-group on the device before readback — writing a partial
per work-item would put megabytes of transfer inside the timed region and
corrupt the very PCIe measurement the GPU work exists to make.

### Saturating the device

Getting a peak number out of a GPU takes more than a correct kernel. Three
things are decoupled deliberately:

| Knob | Chosen by | Why it is separate |
|---|---|---|
| corpus size | `--working-set-kb` | serves the memory axis |
| launch geometry | measured at init | a device property, not a workload one |
| corpus sweeps per launch | measured at init | amplifies work without growing the footprint |

Each work-item strides over as many groups as it takes to cover the corpus, so
the launch is sized for the device while the corpus stays free. Sweep count is
forced **odd**, so XORing every sweep leaves the single-sweep checksum intact —
an even count would cancel to zero and silently weaken verification.

Throughput was previously a function of `--working-set-kb`, which meant the
memory axis and the occupancy axis were the same knob and the default
under-reported the card several-fold (before and
after). It is now flat from about 4 MiB up, and the
reported `kernel busy` fraction shows how much wall time was actually spent
executing rather than launching:

```
  242.00 MH/s   (median of 6 samples, 1 thread)
  launch      6144 work-items x 64 per group, 217 corpus sweeps
  kernel busy 98.9% of wall time
```

**One honest limit:** below roughly 4 MiB there is simply not enough independent
work to fill the device — 768 messages is 12 groups, and no launch geometry
turns that into GPU-scale parallelism. Sweeps amplify total work but not
parallelism.

The integrated GPU beats every CPU kernel on this chip by a wide margin —
by roughly 3.5x.

### When a device number is not plausible

Three checks worth making before believing a GPU result:

- **`verified` false anywhere.** Stop and investigate; nothing else matters.
- **SHA-512 not dramatically slower than MD5.** Consumer GPU ALUs are 32-bit and
  emulate 64-bit integer work, so SHA-512 should fall off a cliff relative to
  MD5 — it does on the iGPU. If a device shows no
  such penalty, check that the kernel is really doing 64-bit work.
- **A link rate well under the interface's capability.** PCIe 4.0 x16 from
  pageable memory should comfortably exceed 5 GB/s. A x8 negotiation or a gen3
  fallback halves your bandwidth and moves N\*, and it is worth confirming the
  negotiated width rather than assuming it:

  ```bash
  nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
  ```

## Energy

Throughput alone does not decide hardware purchases; energy per unit of work
usually does. Where counters are readable, runs report joules, watts and
**hashes per joule**:

```
  Energy
    cpu          10.43 J    21.15 W
    gpu           1.75 J     3.55 W
    efficiency  9772.71 kH/J
```

On the N100 the integrated GPU wins on both counts, and by more on efficiency
than a throughput comparison alone would suggest:
hashes per joule per kernel. RAPL domains cross-check as
they should, package bounding core plus uncore.

Sources, all optional and all discovered at runtime:

| Source | Covers | Notes |
|---|---|---|
| powercap RAPL | CPU package, cores | on client Intel parts the `uncore` domain **is** the integrated GPU |
| DRM hwmon | discrete AMD, Intel | `energy1_input`, or `power1_average` as a fallback |
| NVML (`dlopen`) | NVIDIA | `nvmlDeviceGetTotalEnergyConsumption`, Volta and later |

Energy counters are preferred over instantaneous wattage; where only average
power exists the source name says so, because integrating a sampled watt figure
over a short run is much less trustworthy than reading a counter.

**RAPL is root-only on most modern distributions** (hardening against the
PLATYPUS side channel), so CPU energy is frequently unavailable to an
unprivileged run. That is reported, not worked around:

```json
"energy": {
  "available": false,
  "reason": "RAPL counters exist but are not readable by this user ..."
}
```

To enable it, either run the benchmark as root, or make the counters readable:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

**`setfacl` does not work here** — sysfs has no ACL support, so it fails with
`Operation not supported`. (It *does* work for `/dev/dri/render*`, which lives
on devtmpfs.) The mode also resets on reboot, since sysfs is rebuilt each time;
for persistence use a udev rule, as udev's `MODE=` applies only to device nodes
and not to sysfs attributes:

```
# /etc/udev/rules.d/99-rapl-readable.rules
SUBSYSTEM=="powercap", ACTION=="add", \
  RUN+="/bin/sh -c 'chmod a+r /sys%p/energy_uj 2>/dev/null || true'"
```

## Threading

`--threads N`, defaulting to one per CPU the process is allowed on -- under
`taskset`, a cpuset or a container that is fewer than the machine has, and
oversubscribing needs an explicit `--threads`. The verified batch is
partitioned across threads; each holds the reference checksum for its own slice
and verifies it every rep, so nothing synchronises in the hot path. Because XOR
is associative, the partials recombine to exactly the single-threaded value —
**the checksum is identical at every thread count**, so it stays a valid
fingerprint rather than becoming a per-configuration artifact.

The driving thread is itself worker 0 and does a slice inline. An earlier version
had it merely wait at a barrier, which left N workers plus an idle driver
competing for N cores and pushed the coefficient of variation past 25%.

**On an SMT machine, use either one thread per core or every thread — not a
count in between.** Workers are pinned to the allowed CPUs in ascending order,
and Linux numbers the physical cores before their siblings, so a count between
the two fills some cores twice and leaves others single. The corpus is split
equally regardless, so the doubled cores become stragglers and the whole batch
waits on them. On a 16-core, 32-thread desktop Zen 5 part the seventeenth
thread costs **21%** against sixteen, and the figure does not recover to its
sixteen-thread value until about twenty-four. Full occupancy is worth 16% over
one-thread-per-core, so SMT does pay — but only once every core is loaded
symmetrically. This is the same effect the equal split has across a
heterogeneous set of devices, described under "Multiple devices run
concurrently" above.

## Verification

A benchmark that reports a fast wrong answer has reported nothing. Every kernel
XORs together each digest it computes, and that checksum is:

1. **compared against the scalar reference before timing** — the reference is
   itself checked against the published known-answer vectors (RFC 1321 for MD5,
   FIPS 180-4 for SHA-1 and SHA-512) plus lengths bracketing every block-count
   transition, where padding implementations actually break;
2. **recompared on every timed iteration** — because a startup-only self-test
   cannot catch hardware that is correct when cold and wrong under sustained
   load, which is the failure mode that matters when benchmarking near thermal
   or power limits.

Verification failure is fatal: no performance number is printed.

### What the coefficient of variation does not cover

`cov_percent` is computed over the samples of **one process**, which share a
corpus placement, a thermal state and a boost state. It is a good measure of
whether a run was internally steady and a poor measure of whether the number
will come back the same next time.

The gap can be large. On a desktop Zen 5 part, twelve consecutive
single-thread runs at the default working set — sitting on the L2 boundary
described under "Working set" — spanned 376.6 to 453.1 MH/s, a run-to-run
coefficient of variation of **7.13%**, while each individual run reported
about **0.084%**. Moving the corpus clear of the boundary brought run-to-run
variation to 0.16%, and the two figures then agreed.

So a low `cov_percent` is necessary and not sufficient. Where a number carries
a decision, **repeat the whole process and use the spread between runs as the
error bar**; `tools/compare.py` already judges a delta against the noise both
runs reported, and that noise is the within-run kind.

Because XOR is commutative and associative, the checksum does not depend on how
work is spread across lanes, streams, threads or devices. Every kernel on every
machine must produce the same value, which makes it a portable fingerprint:

```
$ ./build/valubench --json --kernel scalar-s1 --threads 1 | grep checksum
    "checksum": "4634a0fbf02488f992251744c0a8f10b",
$ ./build/valubench --json --kernel avx2-s4   --threads 4 | grep checksum
    "checksum": "4634a0fbf02488f992251744c0a8f10b",
```

It is a fingerprint *within* a workload id, not across one: different message
lengths or iteration counts hash different bytes and so produce different
checksums by design.

## Output

Human-readable by default, `--json` for machines. Both render from the same
structs, so they cannot disagree. The JSON carries the unit and the
direction-of-goodness explicitly, the full sample distribution, the environment,
and any conditions that make the result less trustworthy:

```json
{
  "result": {
    "unit": "H/s",
    "direction": "higher_is_better",
    "median": 54120000,
    "cov_percent": 3.15,
    "stable": true,
    "samples": [ ... ]
  },
  "warnings": ["CPU governor is 'powersave', not 'performance'; ..."]
}
```

Exit status: `0` success, `1` verification failure, `2` usage error, `3` result
too noisy to trust.

## Kernels

One binary contains every ISA path; selection happens at runtime via CPUID.
There is deliberately no `-march=native` — it would make the binary's behaviour
depend on the machine that compiled it, which is exactly the hidden variable that
makes benchmark results incomparable.

```
$ ./build/valubench --list
NAME         ISA       LANES  STREAMS  AVAILABLE
scalar-s1    scalar        1        1  yes
...
avx2-s4      AVX2          8        4  yes
avx512-s4    AVX512       16        4  no
```

**Streams matter as much as lanes.** MD5's 64 steps form a single serial
dependency chain, and SIMD width does not break it — all lanes of a vector
advance in lockstep as one chain. A one-stream kernel measures dependency
latency, not throughput. Interleaving independent chains is what fills the
pipeline, so every ISA is instantiated at 1–4 streams and the harness picks the
winner by measurement rather than assumption:

```
$ ./build/valubench --verbose
Autotune:
  scalar-s1          ...
  scalar-s4          ...
  sse2-s3            ...
  avx2-s4            ...
```

Interleaving alone, with no change of instruction set, is worth **1.70x** on a
genuinely scalar path on an Intel N100 — 9.36 MH/s at one stream against 15.92
at three. It peaks at three there and falls back at four, because four streams
of MD5 state stop fitting in sixteen general-purpose registers.

An earlier figure of 3.1x for this was wrong: the compiler was vectorising the
scalar kernel, so the comparison was scalar against SSE2 rather than one stream
against three.

### The one kernel where streams mostly do not help

`--algorithm sha1` also builds a **SHA-NI** path, which exists to answer a
question the rest of the benchmark deliberately avoids: what is a fixed-function
hash unit actually worth?

`SHA1RNDS4` performs four real SHA-1 rounds in one instruction, so the register
holds *one* message's state rather than a vector of messages — `lanes = 1`. It
runs at more than twice the best SIMD path on the N100, and the stream ordering
changes with it: where every other kernel gains substantially from
interleaving, a lone dependency chain already comes close to saturating the
unit, so extra streams buy little and cost registers.

**How little, and whether "little" means "nothing", is a property of the
part.** On all three Intel parts in the database the ordering is monotonic and
one stream wins. On both Zen 5 parts it is not: throughput dips at two streams,
peaks at three, and only then falls away. That is why the harness measures
instead of applying a rule — a rule learned from one vendor picks a variant
about 20% off the best on the other.

**Read the SHA-NI figure as a ratio, never as an integer-SIMD number.** It is
also flattered by this CPU: Gracemont is an E-core with a 128-bit vector
datapath, so the AVX2 side of the comparison is about half as strong as it would
be on a P-core.

### Streams on a GPU are not streams on a CPU

The same knob behaves in the opposite direction on the device, and it is worth
knowing before porting a CPU kernel — SHA-1 loses an order of magnitude at four
streams and SHA-512 halves at two
(measured).

A GPU already has thousands of work-items in flight, so a second stream hides no
latency that was not already hidden — it only adds live registers. SHA-1 and
SHA-512 both *expand* their sixteen message words rather than permuting them, so
the schedule window must stay resident: sixteen words per stream for SHA-1,
sixteen 64-bit words for SHA-512. Past the register budget the data spills and
throughput falls off a cliff rather than tapering.

MD5 does not have this problem — it permutes sixteen words and can read them
from global memory as each step needs them, which is why `md5/ocl` varies
smoothly with stream count and the other two do not.

## Finding the PCIe crossover

The question the GPU path exists to answer is *"how many iterations of the
kernel do I need to keep the computation GPU-limited rather than PCIe-limited?"*
— because an accelerator that spends its time waiting on the bus is not worth
paying for.

By default the corpus is uploaded once and every launch runs against resident
data, which is the right model for work that lives on the accelerator but means
there is no link traffic to be limited by. `--transfer stream` re-uploads before
every launch and reports the two halves separately:

```
$ valubench --kernel md5/ocl-s1 --working-set-kb 65536 --transfer stream
  kernel busy 20.7% of wall time
  transfer    39.3% of wall, 7.62 GB/s host->device (64.0 MiB per pass)
  bound by    TRANSFER  (compute/transfer = 0.53)
```

Sweep `--iterations` and the ratio walks through 1.0. That crossing is N\*:

```
$ ./tools/sweep.py --algorithm md5 --kernel md5/ocl-s1 --transfer stream \
      --message-bytes 64 --working-set-kb 262144 --iterations 1:1024:*2
```

The measured sweep walks from transfer-bound to
compute-bound within the first few iterations, with the link rate flat
throughout — which is the check that the link is being measured consistently
rather than varying with the workload.

**The balance point is solved, not bracketed.** A geometric sweep only answers
to within its own step. But transfer is constant in the iteration count and
kernel time is linear in it, so `sweep.py` fits both and reports the intercept
(what that looks like).

**Check `r2` before quoting N\*.** It has been 1.0000 on every sweep so far, so
anything below about 0.98 means the linear model broke — the corpus fell out of
a cache partway up the sweep, or the launches were too short to time. The tool
flags that, and flags a transfer rate that moved more than 15%. A fit that did
not hold is not a balance point.

**Use `--message-bytes 64` when comparing algorithms across the link.** It is
the one length where all three store exactly 128 bytes per padded message — MD5
and SHA-1 spill into a second 64-byte block, SHA-512 still fits one 128-byte
block — so the transfer side is identical across the three and only compute
varies. It is also the shortest length SHA-512 can iterate at, since the digest
is fed back over the head of the message.

N* is a property of the (kernel, working set) pair rather than of the kernel:
doubling the corpus moved it measurably, because compute scales linearly while
the achieved link rate does not (both points). Quote the
working set with it.

**The ratio answers the question for a pipelined implementation too**, even
though this one uploads and computes in sequence — overlapping the two can hide
the smaller but never the larger, so whichever side exceeds 1.0 binds either
way. That is why the two times are reported separately rather than folded into
one number.

### Break-even: does the accelerator beat the CPU you already own?

N\* answers "is the bus in the way". It does not answer "should I buy this",
and the two disagree routinely — a device can be compute-limited and still
slower end to end than the host, or bus-limited and still faster. Break-even is
the iteration count at which offloading overtakes the *whole* CPU, transfers
included:

```
$ ./tools/sweep.py --algorithm md5 --transfer stream --where device,cpu \
      --message-bytes 64 --working-set-kb 32768 --iterations 1:16:*2

CPU break-even  (device with transfers == whole CPU)
  md5 64-byte          break-even at ... iterations
      device md5/ocl-s1, CPU md5/avx2-s3 on 4 threads
      per iteration: ... device, ... CPU  (...x compute advantage)
```

`--where cpu` and `--where device` restrict autotune to one side, which is what
makes the comparison possible: autotune otherwise ranks device and CPU kernels
together, so on a machine with a GPU there is no way to ask for the best CPU
kernel — and a "CPU baseline" measured without it is quietly a GPU number.

Both sides are linear in the iteration count, so break-even is solved rather
than searched, exactly like N\*. If the device computes an iteration no faster
than the CPU does, the curves never cross and the tool says so instead of
extrapolating. Measured on the development box, where
the answer is sobering: against four cores rather than one, the iGPU's compute
advantage collapses from 5.15x to roughly 1.6x, and offload only pays at all
after a few iterations have amortised the upload.

Two caveats. The figures above are from an integrated GPU, where there is no
PCIe at all and the "transfer" is a copy within system RAM; the number that
carries a purchasing decision needs a discrete card. And streaming uploads from
pageable host memory, which is honest for "my data is in ordinary memory" but
about half what pinned staging achieves, so N\* should eventually be quoted as a
pair.

## Capturing a whole session

`tools/run.sh` captures the environment, gates on `make check`, runs the matrix,
and leaves one tarball. Seven CPU phases — the ISA ladder, downclocking, energy,
the SHA unit, all three algorithms, stream interleaving, and the two workload
axes — then four device phases if the machine has an OpenCL device: the PCIe
crossover, device stream counts, the memory axis, and multi-device slicing.
Phases that need hardware the machine lacks skip themselves and say so.

```
$ ./tools/run.sh                # everything this machine can do
$ ./tools/run.sh -q             # quick pass
$ ./tools/run.sh --only device  # on a metered GPU box, the expensive half first
```

CONTRIBUTING.md covers the setup failures worth pre-empting on a rented
machine.

If the session is cut short, the phases that finished are the ones that
mattered most.
