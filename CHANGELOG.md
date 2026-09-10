# Changelog

Notable changes. Measured figures are not recorded here; they are tracked
outside this repository while a durable format for them is decided.

## 0.7.0 — 2026-09-09

### Fixed

- **A heap overflow in the correctness oracle, reachable from a documented
  flag.** The digest-fits-message guard tested `cfg.iterations`, which
  `--reference-ladder` never sets, so the ladder walked to its top rung with a
  message shorter than the digest and the feedback `memcpy` wrote past the end
  of its allocation. `--algorithm sha512 --reference-ladder 1,2` at the default
  55-byte message writes 64 bytes into 55, confirmed by AddressSanitizer; the
  guard also ran before the ladder was parsed, so it could not have seen the
  rungs in any case. Both are fixed and four cases are in the output contract.

- **`vb_config_defaults()` left `have_expected` indeterminate.** It assigned
  fifteen fields by name and touched nothing else, and `main()` puts the
  struct on the stack. Non-zero there makes a run skip the reference
  computation and compare against `expected`, garbage from the same frame, so
  a correct kernel reports VERIFICATION FAILED -- non-deterministically,
  varying with compiler and optimisation level. It now zeroes first.

- **Two GPUs from different vendors: one was silently dropped.** VB-004 stopped
  a card being counted twice by picking one energy provider per scope and
  discarding the rest. That is right when two providers see one card and wrong
  when they see two: an Intel iGPU through RAPL uncore beside an NVIDIA card
  through NVML reported **5 J against a true 255**, attributing the compute
  card energy to an idle one, in an ordinary desktop configuration. An AMD card
  beside an NVIDIA card lost the NVIDIA one the same way.

  Deduplication now keys on the device rather than the provider, using the PCI
  address -- from the sysfs symlink for DRM sources, from NVML busIdLegacy for
  NVIDIA ones. Sources that do not identify themselves are summed rather than
  dropped: if they are two devices the sum is right, and if they are one it
  overstates by at most 2x, where dropping understates by everything that
  device was doing.

  No captured result is affected. Every machine measured so far has had exactly
  one GPU energy provider, and the fault needs two.

- **`isa_available` described x86 only, so every AArch64 result named no
  instruction set.** It carried `sse2`, `avx2` and `avx512f` and nothing else,
  which reported all three false on ARM and left the ISA that actually ran
  unrecorded. Cross-machine claims are sourced from exactly that field, and
  986 rows in the results database were affected. All seven flags are now
  reported.

- **RAPL `psys` was counted twice.** `psys` is the whole platform and contains
  the package, but fell to the `else` branch marked as not-contained, so a
  total holding both added the package to itself. `dram` lands in the same
  scope and is genuinely outside the package, so it still sums; only `psys` is
  marked contained.

- **`pin_failed` was read before the workers could set it.** `pool_create()`
  returned and the flag was tested immediately, while workers write it after
  clearing the start gate — so only the driving thread's own pin failure was
  ever visible. It is now read after `calibrate_reps()`, whose barrier pass
  guarantees every worker has run.

- **A dependency line became the Makefile's default goal.** The heap-overflow
  fix placed `$(BUILD)/registry.o: $(KHDRS)` above `all:`, making it the first
  target: `make` built one object and exited 0. CI's build step passed in zero
  seconds and a later job died on a missing binary. The rule has moved down
  beside the other object rules, `.DEFAULT_GOAL := all` is pinned before any
  target so the next rule added above `all` cannot repeat it, and every CI
  build step now asserts the artifact exists rather than trusting the exit
  status. This project has produced a build-that-builds-nothing four times.

- **The AArch64 cross job failed on a noisy run.** Adding `set -o pipefail` so
  a crashed binary would report as a crash also made exit 3 fatal — and exit 3
  is "too noisy to trust", which is ordinary under emulation on a shared
  runner. The step now runs and parses separately, accepting 0 or 3 and
  failing anything else.

- **Documentation claims that measurement has since contradicted.** All found
  by running the benchmark on a desktop Zen 5 part; see the
  `ryzen9950x-20260909` capture.

  `guide.md` still said MD5 "barely becomes memory-bound" and that reaching a
  memory-bound regime would need much larger messages or a bandwidth-hungry
  companion kernel. `design.md` §2a had already retracted that, so the two
  documents disagreed. Whether the workload becomes memory-bound is a property
  of the kernel rather than of MD5: a slow kernel never reaches the roofline
  ridge and a fast one does, and at full core occupancy on a wide part the
  corpus leaving cache is a cliff rather than a knee.

  `guide.md` also stated that SHA-NI is monotonic in stream count — fewer is
  better. That holds on every Intel part measured and on neither AMD part:
  both dip at two streams and peak at three. Applying the Intel rule there
  costs a few percent, and applying the general interleaving rule costs
  substantially more.

  The README listed "SVE and SVE2 have no kernel" as the largest known gap.
  Those kernels exist, are validated on Neoverse V1 and V2, and were missing
  from the ISA list beside it. Its SHA-NI range and its architecture and
  fingerprint rosters were also a capture or more out of date, and its
  toolchain note quoted a two-compiler spread far narrower than the
  four-compiler figure the project has since measured.

  `avx512.c` still carried "UNVERIFIED ON HARDWARE — this path compiles and is
  disassembled but has not been executed". It has been executed on three
  parts; the header now records what the ratio against AVX2 actually is and
  why it differs between them.

### Added

- **A queryable view over the captures.** `tools/ingest.py` builds a SQLite
  view of every capture CSV, stdlib only, in about a second, and
  `docs/results.md` explains why it stays derived and disposable: the CSVs are
  the record, and a bug in the ingest costs a rebuild rather than a
  measurement. Machine identity comes from the capture directory rather than
  the reported CPU, because every ARM part here identifies as "AArch64
  implementer 0x41" and grouping by that silently merged three
  microarchitectures. The ingest names every file it skips and why, since a
  capture script that quietly stopped emitting the sweep schema looks
  identical to one that was never run.

- **`virtualized`, as three states rather than a boolean.** Nothing in a result
  said whether it came from a VM, and every ARM number in this project did.
  The x86 hypervisor CPUID bit is definitive where it exists and has no
  AArch64 equivalent, so a flag derived from it would confidently report "bare
  metal" for machines that are certainly not — the exact failure this
  benchmark exists to avoid. DMI covers both and usually names the hypervisor;
  what neither settles stays `unknown`. The verdict ships with its evidence,
  `sys_vendor` and `product_name`, so a reader can disagree with it.

- **The environment is sampled again after the timed region, and temperature
  is recorded.** Frequency, governor and load were read once at startup,
  describing a machine that had not yet run the benchmark; they are now also
  read afterwards as `*_at_end`, with a warning when the clock falls more than
  5%. Package temperature is recorded at both ends with a warning on a rise of
  more than 10 C, because a part benchmarked cold and one benchmarked after
  twenty minutes of load are different machines. Only CPU and package thermal
  zones are believed, and `temp_source` names what answered.

- **Guidance for three ways a run can report a real number that means
  something other than it appears to**, all in `guide.md`, with the
  `cov_percent` scope note also in `docs/schema.md`.

  The default working set lands on the L2 capacity boundary of a core with
  1 MiB of L2, and what that costs depends on the stream count — the
  high-stream AVX-512 kernels lose heavily where the four-stream one and AVX2
  barely move. The stream ordering inverts across that boundary, so
  **autotune's choice of kernel depends on `--working-set-kb`**.

  A kernel processes `lanes x streams` messages per group, so a pool wants at
  least `threads x lanes x streams` messages before any lane-level figure
  means anything. At long message lengths the default corpus supplies far
  fewer and the point silently under-reports. This is the CPU form of the
  device-side saturation limit already documented.

  `cov_percent` is within-process dispersion, not reproducibility: its samples
  share one corpus placement, one thermal state and one boost state. Measured
  on a desktop Zen 5 part, consecutive runs of one command spanned two orders
  of magnitude more variation than any single run reported. A low
  `cov_percent` is necessary and not sufficient.

- **A note that partial SMT occupancy is pathological**, in `guide.md`.
  Workers pin in ascending CPU order and Linux numbers physical cores before
  their siblings, so a thread count between one-per-core and full occupancy
  fills some cores twice and leaves others single. The corpus splits equally
  regardless, so the doubled cores become stragglers the batch waits on. Use
  one thread per core or every thread, not a count in between.

### Changed

- **Cross-machine reasoning now lives with the captures, not in this
  repository.** Conclusions that span more than one machine — each stated with
  the SQL that produces it — sit beside the results they interpret, for the
  same reason the measurements do: a claim is false the moment its query stops
  supporting it, and that should not be a commit against the source tree.
  Single-machine results stay in their own capture's README.
  `docs/results.md` records the split.

## 0.6.0 — 2026-08-28

### Fixed

- **Autotune pinned every worker to one core, under-reporting by up to 3.4x.**
  `pool_create()` pins worker 0 by calling `pthread_setaffinity_np` on the
  calling thread, and never restores it. `vb_allowed_cpus()` then read the mask
  back with `sched_getaffinity(0, ...)`, which reports the *calling thread's*
  mask rather than the process's. Autotune builds one pool per candidate kernel
  on that same thread, so the first probe narrowed the mask to a single CPU and
  every pool after it — including the one behind the reported result — put all
  of its workers there.

  Nothing gave it away. `threads_used` still reported the count that was asked
  for, and no pin failure was raised, because pinning to a CPU you already
  occupy always succeeds. Measured at 2.9x slow on four cores and 3.4x on
  eight, and confirmed with `ps -L`: every thread on PSR 0 before the fix,
  spread across all CPUs after.

  It also chose the wrong kernel. Every candidate was ranked on one core, so
  autotune selected the kernel that wins under contention rather than the one
  that wins on the machine.

  **No released version is affected.** The defect was introduced after 0.5.0
  was tagged, by the VB-007 pinning fix on 2026-08-25, and never appeared in a
  release — 0.5.0 derives worker CPUs from the online count and contains no
  `sched_getaffinity` call at all. It affects builds taken from `main` between
  2026-08-25 and this release.

  For such a build, a figure is suspect only if all three hold: it ran on the
  CPU, with more than one thread, and let autotune pick the kernel. Naming
  `--kernel` means one pool per process and the mask is read before the first
  pin can narrow it.

  Results now carry `pinned_cpus` beside `threads_used`. The broken build
  reports `threads_used=8, pinned_cpus=1`; nothing in the old output could
  express that, which is how this passed review, CI, and five hardware
  sessions.

- **The binary died at startup on every AArch64 CPU without SVE.** The registry
  resolved lane counts by calling each kernel's `lanes_fn` with no
  `available()` check, and for SVE rows that function is a bare `CNTW` — an SVE
  instruction, UNDEF without the feature. SIGILL on Graviton2, Ampere Altra,
  every Raspberry Pi and every Cortex-A5x/A7x, before computing anything.

  Four parts, because guarding the call alone trades a crash for a lie:
  `resolve_lanes()` consults `available()`; `vb_batch_divides()` returns 0 for a
  zero group size, since AArch64 `UDIV` by zero yields 0 rather than trapping
  and `768 % 0` evaluated to 768; the structural test skips rows whose width
  needs an absent ISA; and `vb_cpu_has_sve2()` now requires SVE, because qemu's
  `-cpu max,sve=off` clears `HWCAP_SVE` while leaving `HWCAP2_SVE2` set and
  that reached `CNTW` by a separate path.

  CI now runs the binary on `neoverse-n1`, `cortex-a72`, `cortex-a53` and
  `max,sve=off`, and asserts the SVE rows stay registered-but-unavailable so a
  build that dropped them cannot pass by testing nothing.

- **Multi-block messages were corrupted on SVE.** An unbraced `if (b == 0)`
  guarded one of five macro-expanded statements. Invisible at the default
  55-byte message, which is a single block; found on real hardware at 112 of
  1032 checks.

- **Sixteen findings from an independent review**, four of which were verified
  against the code before being accepted and two of which were materially worse
  than filed. A degraded thread pool reported **176 MH/s against a true 43** and
  marked it verified, because slices were sized before any thread started while
  `hashes_per_iter` counted the whole corpus; pools are now all-or-nothing
  behind a start gate. An energy total counted Intel's `uncore` domain twice,
  understating hashes/joule by 15% on client parts.

- **`make check` failed on any SVE machine whose vector length is not a power
  of two** — 12 failures at 384 bits, blaming kernels that were correct. A
  run-time lane count that does not tile the batch is a property of the machine;
  a fixed-width one that does not is a defect, and only the second now fails.

- **A hang counted as a passing test.** `check-threadfail` asserted that a
  degraded pool never reports a result, with no time bound — and removing the
  start gate to verify that assertion deadlocks rather than returning a wrong
  number. Two such waits sat on a development machine for 9 and 23 hours. Each
  fault injection is now bounded.

- **A guard that found nothing passed.** The scalar-purity check reported
  success while inspecting zero kernels, and immediately caught a real
  `OBJDUMP` derivation bug once it was made to fail on an empty set.

- **SHA-512 reported half its working set.** The result path computed corpus
  size with a hardcoded 64-byte block, which is right for MD5 and SHA-1 and half
  the truth for SHA-512's 128-byte blocks. The corpus was always *built* at the
  correct size, so no throughput figure was wrong — but working set is one of
  the three axes this benchmark sweeps, and the reported number decides which
  side of a cache a measurement is read as sitting on. It now comes from the
  algorithm, and `make check` verifies the reported figure against
  `batch_messages x blocks x block_bytes` for every algorithm at each padding
  boundary.
  A second, correct implementation of the same arithmetic existed in
  `vb_working_set_bytes()` and had no callers, which is how the two could
  disagree unnoticed; it has been removed rather than wired up, because the
  reported figure should describe the corpus that was built rather than one
  recomputed from the request.
  **Comparing SHA-512 results across this fix**: `tools/compare.py` treats
  working set as part of a measurement's identity, so a pre-fix SHA-512 row will
  not pair with a post-fix one even when the runs were otherwise identical. That
  is the tool being right — the two labels genuinely differ — but it means
  archived SHA-512 comparisons need the older side's figure doubled first.
- **The scalar kernel was not scalar.** At `-O2`, GCC's SLP vectoriser fused the
  independent streams and emitted SSE2 on x86-64 and NEON on AArch64 — 88% and
  79% of the two-stream kernel's instructions — while clang did not, so the
  baseline every ISA ratio divides by depended on the compiler. That translation
  unit is now built with `-fno-tree-vectorize -fno-tree-slp-vectorize`, and
  `make check` disassembles the result and fails if more than 5% of its
  instructions touch a vector register. Any scalar figure measured before this
  is not a scalar baseline and must not be used as a denominator.

### Added

- **SVE and SVE2 kernels, vector-length agnostic.** One binary correct at 128,
  256, 384, 512, 1024 and 2048-bit vector lengths, verified at every one. SVE
  types are sizeless and cannot go in the arrays the shared templates use, so
  the templates gained hooks whose defaults are the original code — the x86
  objects are byte-identical with and without them, checked per symbol.

  SVE-256 beats NEON-128 on Neoverse V1 by 1.12-1.20x *with a compiler that can
  generate it*; gcc 13 and 14 emit as many instructions for eight lanes as NEON
  needs for four, which is a code-generation defect fixed in gcc 15. The
  toolchain is worth up to 4.15x on the same silicon — more than any
  architectural difference this project has measured.

- **GPU clock, temperature and throttle-reason telemetry**, sampled across the
  timed region and reported as first/last/min/max. NVML only; parts without it
  report nothing rather than zero. This is what the word *sustained* rests on:
  every device figure before it was potentially a boost-clock number, and no
  sustain mode was needed because `--warmup-ms` already accepts up to an hour.

- **Stream counts beyond four** — 6 and 8 — which MD5 wanted, and which are
  worth 26-36% on Neoverse V2.

- **`pinned_cpus` in every result**, so a pool confined to fewer cores than it
  claims says so on the face of the output.

- **`check-pinning`**, which asserts a multi-threaded pool spans more than one
  CPU. This is the check that was missing when autotune collapsed every worker
  onto one core: no baseline is needed, because it is an invariant rather than
  a comparison against a recorded figure — and a recorded figure was never
  available, since results are deliberately kept out of the repository. It
  fails on a build with the bug, and fails again if `pinned_cpus` ever
  disappears from the output rather than passing quietly. A single-CPU machine
  says it cannot run the check; CI asserts the runner has more than one core,
  so a change there cannot make it vacuous.

- **Virtual or bare metal**, recorded as `virtualized: yes | no | unknown` with
  the DMI evidence behind it. Three states rather than a flag because on
  AArch64 the x86 hypervisor CPUID bit has no equivalent: `lscpu` reports no
  hypervisor on Graviton and Grace alike, machines that are certainly virtual,
  so a boolean built on that evidence would have labelled every ARM figure in
  this project bare metal. Sixteen constructed cases cover the shapes no one
  machine has, including the EC2 pair that names itself identically whether
  virtual or not and is told apart only by its instance type.

- **CI gained teeth**: an ASan/UBSan job, a software-OpenCL job, SVE at four
  vector lengths under emulation with a deliberate non-power-of-two width, a
  multi-block sweep against the scalar reference, an objdump assertion that the
  SVE units contain SVE, and the non-SVE AArch64 startup check above.

- **`run.sh` gained D5**, the resident iteration ladder: where compute overtakes
  *memory*, as distinct from D1's where it overtakes the link. Two message sizes,
  because the knee is compressions per byte read and `blocks_per_message` is the
  other half of that ratio.

### Changed

- **Relicensed to BSD 3-Clause**, with SPDX identifiers on every source file.
- **Measured results moved out of the repository.** `RESULTS.md` worked for one
  machine and stopped working at five: numbers outlived their provenance, and
  nothing was machine-checkable.
- **The iteration ladder is solved in one pass** rather than once per rung,
  which took reference computation from dominating a crossover session to
  disappearing into it.
- **`tools/isa_cost.py` no longer claims to predict throughput.** It supplies
  instructions per hash; throughput is that times IPC, and its predictions were
  wrong in both directions.

## 0.5.0 — 2026-08-23

First public release. What it contains:

### The benchmark

- **MD5, SHA-1 and SHA-512**, each written from its specification rather than
  adapted from an existing implementation, which is what leaves the whole tree
  free of any third-party notice.
- **A compile-time kernel matrix selected at runtime**: scalar, SSE2, AVX2,
  AVX-512 and SHA-NI on x86, NEON on AArch64, OpenCL on a device, each at one to
  four interleaved streams. One binary carries every path, `CPUID` and
  `getauxval(AT_HWCAP)` choose between them, and the choice is reported in the
  output. No `-march=native`, so a binary's behaviour never depends on the
  machine that built it.
- **Autotune**, which measures the available kernels rather than assuming. It
  has to: the best stream count varies by algorithm and by core, and a
  fixed-function SHA unit is 2.13x faster than the vector path on one
  microarchitecture and 0.46x as fast on another.
- **`--where cpu|device|any`** to restrict autotune to one side, without which a
  CPU baseline on a machine with a GPU is quietly a GPU number.

### Correctness

- Every kernel is checked against a scalar reference, and each run reduces its
  digests to an XOR fingerprint **invariant across lanes, streams, threads,
  devices and instruction set architectures**. The same three checksums come
  back from Gracemont, Cascade Lake, Zen 5 and Neoverse V1.
- `make check` runs 1468 kernel and known-answer checks. CI builds with gcc and
  clang and cross-compiles for aarch64 to run the NEON path under emulation, so
  the ARM kernels cannot rot while nobody has the hardware.
- A run that fails verification produces no number.

### Measurement

- **JSON and human-readable output rendered from one data model**, so the two
  cannot disagree. `--list --json` dumps what the binary can do on this machine,
  which is how the tooling avoids carrying a transcribed copy that drifts; all
  four schemas are documented in [docs/schema.md](docs/schema.md).
- **Dispersion is reported, not hidden** — min, median, mean, stddev, CoV and
  the raw samples, with a run flagged when its own noise exceeds the
  significance bar.
- **The environment is captured automatically**: CPU model and flags, the ISA
  path actually taken, governor and frequency, thread count, compiler, device
  and driver. A number without its machine is not comparable to anything.
- **Energy per hash** where powercap exposes it, from `intel-rapl` or
  `amd-rapl`.
- **The PCIe balance point and CPU break-even are solved rather than
  bracketed.** Transfer is constant in the iteration count and compute is linear
  in it, so both come from a linear fit, with the fit quality reported alongside.

### Tools

- `tools/sweep.py` — parameter sweeps to CSV, with resume.
- `tools/compare.py` — diffs two result sets, pairing by workload id, refusing
  to compare across a checksum mismatch, and judging a delta against both the
  significance bar and the noise the runs themselves reported.
- `tools/run.sh` — a one-command capture for a time-boxed session on rented
  hardware: CPU phases, then device phases if there is a device, ordered so an
  interrupted session still yields what mattered most.

### Known limits

- **No SVE or SVE2 kernel.** Vector-length-agnostic code does not fit the
  fixed-width kernel template and needs its own.
- **OpenCL is validated on an Intel iGPU only**, which has no PCIe link, so
  every device figure comes from a part where "transfer" is a copy within system
  RAM.
- **No bare-metal run**, leaving AVX-512 licence downclocking and energy
  unmeasured.
- **Device uploads come from pageable memory**, roughly half what pinned staging
  achieves, which moves the balance point by about that factor.
