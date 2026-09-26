<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026, The valubench authors. See LICENSE.
-->

# Working with captured results

Measured numbers live outside this repository. `tools/ingest.py` builds a
queryable database from a directory of them.

    tools/ingest.py ~/projects/valubench-results
    sqlite3 ~/projects/valubench-results/results.db

## What is authoritative and what is not

The CSVs a capture leaves behind are the record. `results.db` is derived from
them, rebuilt in under a second, and safe to delete — nothing should ever read
it as the source of truth, and a bug in the ingest costs a rebuild rather than
a measurement.

This split is the whole design. An earlier `RESULTS.md` inside the repository
mixed a database with an argument: figures were hand-transcribed from CSVs that
already existed, two had drifted into disagreeing copies, and re-measuring meant
rewriting prose. Keeping the derived thing disposable is what stops that.

**The reasoning is kept outside the repository too.** Conclusions that span
more than one machine — each stated with the SQL that produces it — belong with
the captures they interpret rather than with the tool, for the same reason the
numbers do: a claim goes stale when a re-measurement stops supporting it, and
that should not be a commit against the source tree. Single-machine results
belong to their capture's own README, where the conditions that produced them
are recorded beside them.

## Why a database rather than the CSVs

Ninety-odd files across sixteen directories answer a question like *where does
SVE2 beat NEON at equal width* only through a one-off script, and every such
script is a fresh chance to get it wrong. One table answers it in SQL:

```sql
SELECT part, compiler, algorithm,
       MAX(CASE WHEN isa='SVE2' THEN hashes_per_sec END) /
       MAX(CASE WHEN isa='NEON' THEN hashes_per_sec END) AS ratio
FROM m WHERE threads=1 AND status='ok' AND isa IN ('NEON','SVE2')
GROUP BY part, compiler, algorithm HAVING ratio > 1.0 ORDER BY ratio DESC;
```

Other things worth having ready:

```sql
-- best single-thread md5 per machine
SELECT part, kernel, compiler, MAX(hashes_per_sec)/1e6 mhs
FROM m WHERE algorithm='md5' AND threads=1 AND status='ok' AND runs_on='cpu'
GROUP BY part ORDER BY mhs DESC;

-- the checksum invariant, across every machine and instruction set
SELECT working_set_kb, COUNT(DISTINCT checksum) sums, COUNT(DISTINCT part) parts
FROM m WHERE workload='md5-full-55x1' AND status='ok' AND verified=1
GROUP BY working_set_kb;          -- sums must be 1 on every row

-- pools that did not spread: more than one thread on fewer than two CPUs,
-- the assertion check-pinning makes. A row with pinned_cpus 0 was run with
-- --no-pin, which is deliberate; 1 is the collapsed pool that under-reported
-- several-fold on main before 0.6.0. Captures taken before 2026-09-12 have no
-- pinned_cpus column and cannot be checked this way.
SELECT capture, source_file, kernel, threads, pinned_cpus
FROM m WHERE runs_on='cpu' AND threads > 1 AND pinned_cpus < 2;
```

## Machine identity comes from the directory, not the CPU

Every ARM part in this project identifies itself as `AArch64 implementer 0x41`
— Graviton3, Graviton4 and Grace alike — so grouping by the reported CPU
silently merges three microarchitectures across 1300 rows. Identity is derived
from the capture directory name instead, giving `part` and `uarch` columns, and
the reported string is kept beside them as a cross-check.

Capture directories are therefore named `<instance>-<label>-<YYYYMMDD>`, and a
new instance type needs a line in `PARTS` in `tools/ingest.py`. The ingest says
so when it meets a prefix it does not recognise.

## Captures must emit the sweep schema

`tools/sweep.py` writes a CSV that carries compiler, CPU, governor,
version, checksum and CoV on every row. The ingest reads that and nothing else.

A capture script that hand-rolls its own CSV will be skipped — and a skipped
file is indistinguishable from a capture that was never taken. This has already
happened once: a session investigating whether *the compiler* explains a
scaling deficit wrote nine columns with no compiler among them. Use `sweep.py`,
or match its schema.

The ingest prints every file it skipped and why, so the failure is visible
rather than quiet. Most skips are legitimate — `perf` output is not a sweep —
but the list is worth reading when a capture seems thinner than it should be.
