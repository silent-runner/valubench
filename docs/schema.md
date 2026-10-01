# Output schemas

valubench emits JSON so that nothing has to scrape human output. Five documents
are defined, each carrying a `schema` field naming itself and its version.

| schema | produced by | contents |
|---|---|---|
| `valubench/result/2` | `--json` | one measurement |
| `valubench/capabilities/1` | `--list --json` | what this binary can do here |
| `valubench/devices/1` | `--list-devices --json` | the `opencl` object; `backends`, each available or why not; and `devices`, each physical device once with its PCI address and the APIs that reach it |
| `valubench/reference/1` | `--reference-ladder` | expected checksums for a ladder of iteration counts |
| `valubench/comparison/1` | `tools/compare.py --json` | a diff of two result sets |

## The compatibility rule

**Adding a field is compatible. Removing or renaming one is not.** A consumer
should therefore match on the prefix and ignore fields it does not recognise:

```python
if not doc["schema"].startswith("valubench/result/"):
    raise ValueError("not a valubench result")
```

Both bundled tools do exactly that — see `Capabilities.query` in
[../tools/sweep.py](../tools/sweep.py) and `load_file` in
[../tools/compare.py](../tools/compare.py).

The version number moves when a field is removed or its meaning changes.

## `valubench/result/2`

**What changed from `/1`** (0.7.0 and earlier): `energy.hashes_per_joule`
divides by the hardware that hashed rather than by the whole machine -- see
below. Nothing was removed; every other difference is an added field. A `/1`
result's `hashes_per_joule` compares with a `/2` result's
`hashes_per_joule_machine`.

| object | fields |
|---|---|
| `benchmark` | `name`, `version`, `build`, `algorithm`, `workload`, `workload_description`, `comparability` |
| `parameters` | `message_bytes`, `digest_bytes`, `block_bytes`, `blocks_per_message`, `iterations`, `threads`, `batch_messages`, `working_set_bytes` |
| `result` | `unit`, `direction`, `median`, `min`, `max`, `mean`, `stddev`, `cov_percent`, `stable`, `cov_threshold_percent`, `message_bytes_per_second`, `compressions_per_second`, `samples`, `total_hashes`, `total_seconds` |
| `verification` | `verified`, `checksum`, `method` |
| `kernel` | `name`, `isa`, `lanes`, `streams`, `selected_by`, `runs_on` |
| `device` | present only for device kernels: name, vendor, driver, `backend`, `compiler`, `compiler_version`, `compile_mode`, `platform`, `pci_address`, `primitives`, `steers`, `geometry_source`, launch geometry, `concurrent_work_items` and `waves_per_sweep`, `kernel_busy_fraction`, `transfer_mode` (`resident`, `stream` or `overlap`), with `overlap` also `pipeline_chunks` (the count used) and `overlap_efficiency`, in either streaming mode `host_memory` and the per-pass transfer figures, and where NVML answers `gpu_clock_mhz` (`first`, `last`, `min`, `max` SM clock over the timed samples, and `samples`), `gpu_temp_c` and `gpu_throttle_reasons` |
| `environment` | `cpu`, `cpus_online`, `smt_active`, `governor`, frequency fields, `loadavg_1min`, `kernel_version`, `os`, `compiler`, `isa_available`, `threads_used`, `pinned_cpus`, `can_pin`, `virtualized`, temperature fields |
| `energy` | `available`, and either `reason` or the figures: `cpu_package_joules`/`watts`, `gpu_joules`/`watts`, `hashing_joules`/`watts` and `hashes_per_joule` (or `hashing_unmeasured`), `machine_joules` and `hashes_per_joule_machine`, and `sources`, each with `counted` |
| `warnings` | array of strings; conditions that make the number less trustworthy |

Three fields carry more weight than the rest:

**`benchmark.workload`** — e.g. `md5-full-55x1`. Any change to the work per hash
changes this string, so two results are comparable only when it matches. Both
tools refuse to compare across a difference. It contains no product name, so it
survived the rename and older captures still pair correctly.

**`verification.checksum`** — the XOR of every digest computed. Invariant to
lanes, streams, threads and devices, so for a given workload it is the same on
every machine that computes correctly. `compare.py` treats a mismatch as a hard
failure rather than a performance delta: it means one side computed something
else.

**`device`, what ran the kernel.** `backend` is the API (`opencl` or `cuda`);
`compiler` and `compiler_version` say what compiled the program -- for OpenCL
the platform, for CUDA `nvrtc` and its version, and on NVIDIA either way the
NVVM frontend version read from the PTX -- and `compile_mode` how: `driver` for
OpenCL, `ptx-jit` or `cubin` for CUDA. Under `--import-ptx` the compiler is
`imported`, its version names the file and the NVVM that built it, and
`steers` reads `imported`, since they were chosen when the file was built. `steers` lists the primitive spellings compiled in
(`rotl32=ptx`), empty when every primitive was plain C, and `primitives` says
whether steering was on at all. `geometry_source` is `pinned` when
`--device-geometry` chose the launch and `tuned` otherwise; two results are
only comparable as compiled kernels at the same `global_work` and
`local_work`.

**`concurrent_work_items` is how many work-items of this kernel the device
held at once** at the launch's group size, measured by the probe every device
kernel carries, and 0 if it could not be measured. `waves_per_sweep` is the
work-items that had work over that, rounded up: 1 means every repeat of the
sweep re-read the whole corpus, so the working set is the one named. More --
only a pinned grid larger than the device, or an unmeasured capacity -- means
each repeat re-read one wave's share, possibly from a cache the corpus does
not fit, and the result says so; 0 means unknown. Results from before
2026-09-27 lack both fields, and a resident one at a corpus larger than the
device's L2 may be such a result.

**`benchmark.build` says which code produced the result**: `git describe` of
the tree it was built from, so `v0.8.0` for a release, `v0.8.0-3-gabc1234` three
commits after one, a short hash with no tag reachable, `-dirty` for uncommitted
changes, and `unknown` outside a git checkout. `version` names the release
alone, and between releases every build claims the last one; `build` is what
tells them apart. The sweep CSV carries it as `valubench_build`, and
`--list --json` and `--version` report it too. Absent before 0.8.0.

**`energy.hashes_per_joule` divides by the hardware that hashed** -- the
cards the run used, matched by PCI address, and the CPU package when the CPU
hashed -- which is `hashing_joules`, and each source says whether it was
`counted`. Until 2026-09-26 it divided by every counter in the machine; that
figure is now `hashes_per_joule_machine`, so compare an earlier capture's
`hashes_per_joule` with this one's `hashes_per_joule_machine`. When nothing
that hashed was measured, `hashes_per_joule` is absent and
`hashing_unmeasured` says why.

**`environment`, the fields that describe drift rather than state.** Frequency,
governor, load and temperature are sampled twice: once at startup and again
after the timed region, as `*_at_end`. A single startup reading describes a
machine that has not yet run the benchmark, so a part measured cold and one
measured after twenty minutes of load are indistinguishable without the pair.
A clock that fell more than 5%, or a package that warmed more than 10 C, raises
a warning. `-1` means the reading was not available; `temp_source` names the
sensor believed, and only CPU or package zones are — an ambient reading dressed
as a core temperature would be worse than none.

**`environment.can_pin`** — whether this platform has a thread-affinity API at
all. It separates two states that `pinned_cpus: 0` otherwise conflates: a pin
that was attempted and refused, which is a defect, and a platform where there
was nothing to attempt, which is a property. A consumer grouping results by
whether pinning was even possible reads this rather than parsing the warning
text. Always true on Linux; false on macOS, which has no affinity API. The
sweep CSV carries it beside `pinned_cpus`.

**`environment.pinned_cpus`** — the distinct CPUs the worker pool actually
spread across. `threads_used` says how many threads were asked for and answers
nothing about where they ran: a pool reporting eight threads on one core looks
identical in every other field, and did, for three days.

**`environment.virtualized`** — `yes`, `no`, or `unknown`, with `sys_vendor`
and `product_name` as the evidence. Three states because the x86 hypervisor
CPUID bit has no AArch64 equivalent, so its absence there is not evidence of
bare metal.

**`result.stable`** — false when `cov_percent` exceeds the threshold. The process
also exits 3 in that case. The number is still a real measurement; it just
should not carry an argument on its own.

**`result.cov_percent` is within-process dispersion, not reproducibility.** The
samples it summarises share one corpus placement, one thermal state and one
boost state, so it answers "was this run steady" and not "will this number come
back". The two can differ by a large factor — on a desktop Zen 5 part,
consecutive runs of one command varied by nearly two orders of magnitude more
than each run reported about itself, because the default corpus sat on an L2
capacity boundary. A consumer deciding
whether a change is real should compare the spread *between* runs; see
[guide.md](guide.md), "What the coefficient of variation does not cover".

### Exit codes

Reported in the capabilities document under `exit_codes`, so a driver need not
hard-code them: `0` success (`ok`), `1` verification failure
(`verify_failed`), `2` usage error (`usage`), `3` result too noisy to trust
(`noisy`), `4` could not run (`cannot_run`). Exit 3 still produces valid JSON on
stdout.

Each code says what to do next. `1` is only ever a digest that did not match,
whatever `--where` or `--kernel` restricted the candidates to: investigate the
machine. `2` means the command is wrong -- malformed or contradictory options --
and would be wrong anywhere. `4` means the command is valid but this machine,
now, could not run it: a `--kernel` needing an instruction set or device this
machine lacks, `--where device` with no device, a `--device` past the last one,
a corpus that could not be allocated, worker threads that would not all start,
a device that could not be set up. Record it and move on, or retry elsewhere or
smaller. The reason is on stderr. A command both contradictory and impossible
here gets `2`. `cannot_run` is absent from 0.7.0 and earlier, which returned `2`
for all of these.

## `valubench/capabilities/1`

`--list --json` describes the binary and the machine: `benchmark`, `limits`,
`defaults`, `exit_codes`, `transfer_modes`, `host_memory_modes`,
`primitives_modes`, `device_geometry`, `dump_device_code`, `compile_modes`,
`import_ptx`,
`backend_filters`, `where_filters`,
`algorithms` (each with digest and block geometry and its minimum iteration
message length), `kernels` (each with isa, lanes, streams, `where`, and
`available` on this machine), the `opencl` object, and `backends` and `devices`
as `--list-devices --json` gives them.

A default of 0 means "chosen at run time": `threads` one per allowed CPU,
`pipeline_chunks` per device, recorded in each result as the count used.

`limits` gives the range the binary enforces on each numeric flag: message
length, iterations, threads, working set, samples, sample and warm-up time, and
pipeline chunks. A driver can check a request against it before making one;
`sweep.py` refuses a grid outside it before any point runs.

It exists so that tooling asks rather than assumes. Every fact in it was once
transcribed into `sweep.py`, and the transcription drifted — the per-algorithm
minimum message length was carried as a single constant, which silently skipped
every SHA-512 point with more than one iteration.

Unavailable kernels are listed too, with `available: false`. "This build has
AVX-512 kernels but this CPU cannot run them" is a different situation from
"this build has none", and a sweep should be able to tell them apart.

## `valubench/reference/1`

`--reference-ladder 1,2,4,8` emits the expected checksum for each iteration
count, alongside the corpus the values are for: `algorithm`, `message_bytes`,
`working_set_kb`, `start_index` and `count`.

The corpus identity travels with the answers because a checksum only means
anything for the exact range it was computed over. A consumer that reused these
values under a different message size or working set would be verifying against
the wrong truth and would not find out — which is the one way a precomputed
gate can be worse than no gate at all. `sweep.py` keys its cache on all three.

Why the document exists: the digest after *k* iterations is a prefix of the
chain for any larger *k*, so an iteration ladder asks for the same walk over and
over. One pass to the largest count, snapshotting at each rung, replaces the
whole ladder, and the saving is the ladder's sum over its maximum — 2x for a
nine-rung power-of-two ladder, 4x for the fifteen-rung ones a crossover sweep
generates. Feed a rung back with `--expect` and that point skips its own
reference pass entirely.

## CSV from `tools/sweep.py`

One row per measured point, in the column order of `CSV_COLUMNS` in
[../tools/sweep.py](../tools/sweep.py). It is not versioned; `--resume` refuses
to append to a file whose header differs rather than mixing two shapes.

Three columns are worth knowing:

**`point_id`** encodes what was *requested* — algorithm, kernel, where,
transfer, host memory, pipeline chunks, primitives, compile mode, device
geometry, message bytes, iterations, working set, threads. `--resume` matches on
it. It exists because requests cannot be recovered from results: `transfer_mode`
comes back empty for CPU kernels whatever was asked for, and `kernel` comes back
as autotune's choice where the request was "pick one".

**`working_set_kb`** is what was actually used, not what was asked for. The
batch has a floor of `VB_BATCH_LCM` messages, so several small requests round
onto the same corpus — which the sweep warns about, and which is why the
requested value lives in `point_id` instead.

**The energy columns** -- `hashing_watts`, `hashes_per_joule`,
`hashes_per_joule_machine` -- are empty where no counter measured what they
describe; `hashes_per_joule` in particular is empty on a CPU row when the
package counter is unreadable, rather than filled from a card that did not
hash.

**The device columns are empty on CPU rows**, not missing: `device`,
`backend`, `device_compiler`, `device_compiler_version`, `compile_mode`,
`device_driver`, `platform`, `pci_address`, `primitives`, `steers`,
`geometry_source`, `global_work`, `local_work`, `sweeps_per_launch`,
`concurrent_work_items`, `waves_per_sweep`,
`transfer_mode`, `host_memory`, `pipeline_chunks`, `overlap_efficiency`,
`gpu_mhz_min`, `gpu_mhz_max`, `gpu_throttle`, `kernel_busy_pct`,
`transfer_busy_pct`,
`transfer_gbytes_per_sec`, `compute_transfer_ratio` and `bound_by` all describe
a host-to-device transfer, and a CPU kernel reads the corpus out of the memory
it is already running in. The CSV always carries the columns so that CPU and
device sweeps concatenate; the human table drops any column that no row filled,
so a CPU-only sweep does not print two permanently blank ones.
