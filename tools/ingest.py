#!/usr/bin/env python3
"""ingest.py -- build a queryable SQLite view over a directory of captures.

SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026, The valubench authors. See LICENSE.

The CSVs a capture leaves behind are the record; this database is derived from
them and disposable. Delete it and rebuild whenever a capture lands:

    tools/ingest.py ~/projects/valubench-results
    sqlite3 ~/projects/valubench-results/results.db

Why a database rather than the CSVs directly: 92 files across 16 directories
answer a question like "where does SVE2 beat NEON at equal width" only through
a bespoke script, and every such script is a chance to get it wrong. One table
answers it in SQL.

Two things this deliberately does not do. It does not become the record --
nothing here is authoritative, so a bug in it costs a rebuild and never a
measurement. And it does not silently skip: a file it cannot read is reported,
because a capture that quietly fails to land is indistinguishable from one that
was never taken.
"""

import csv
import glob
import os
import re
import sqlite3
import sys

# The CPU a machine reports is not enough to tell machines apart. Every ARM
# part in this project identifies as "AArch64 implementer 0x41" -- Graviton3,
# Graviton4 and Grace alike -- so grouping by it silently merges three
# microarchitectures. The capture directory name already carries the answer,
# so identity comes from there and the reported string stays as a cross-check.
PARTS = [
    ("c7g4xl", "Graviton3", "Neoverse V1"),
    ("c7g",    "Graviton3", "Neoverse V1"),
    ("c8g",    "Graviton4", "Neoverse V2"),
    ("c8a",    "EPYC 9R45", "Zen 5"),
    # Not a cloud instance: a workstation, so the "instance" column carries the
    # part name. Kept in the same table because identity still comes from the
    # directory, which is the property that matters here.
    ("ryzen9950x", "Ryzen 9 9950X", "Zen 5"),
    ("gh200",  "Grace",     "Neoverse V2"),
    ("metal-spr",     "Xeon 8488C", "Sapphire Rapids"),
    ("lambda-2xh100", "Xeon 8480+", "Sapphire Rapids"),
    ("lambda-a10",    "Xeon 8358",  "Ice Lake-SP"),
    ("a100-sxm40",    "EPYC 7J13",  "Milan"),
]

# Columns that are numbers. Everything else is stored as text.
NUM = {
    "message_bytes", "blocks_per_message", "iterations", "threads",
    "working_set_kb", "working_set_bytes", "batch_messages", "lanes",
    "streams", "hashes_per_sec", "hashes_per_sec_min", "compressions_per_sec",
    "message_bytes_per_sec", "cov_percent", "kernel_busy_pct",
    "transfer_busy_pct", "transfer_gbytes_per_sec", "compute_transfer_ratio",
    "kernel_ns_per_pass", "transfer_ns_per_pass", "samples", "loadavg_1min",
    "freq_khz_at_end", "temp_milli_c", "temp_milli_c_at_end", "pinned_cpus",
}
BOOL = {"stable", "verified", "smt_active"}


def identify(capture):
    """(instance, part, uarch) from a capture directory name."""
    for prefix, part, uarch in PARTS:
        if capture.startswith(prefix):
            return prefix, part, uarch
    return capture.split("-")[0], None, None


def coerce(value, column):
    if value is None or value == "":
        return None
    if column in NUM:
        try:
            return float(value)
        except ValueError:
            return None
    if column in BOOL:
        return 1 if str(value).lower() in ("true", "1", "yes") else 0
    return value


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.expanduser(root)
    db_path = os.path.join(root, "results.db")

    files = sorted(glob.glob(os.path.join(root, "*", "*.csv")))
    if not files:
        sys.exit(f"ingest: no capture CSVs under {root}")

    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)

    # The schema is the union over every readable file, not the first one's.
    # Captures accumulate columns over time -- `virtualized` arrived after
    # sixteen captures existed -- and taking the first file's header would
    # drop whatever a later, wider file added, silently and depending on
    # nothing more than sort order.
    columns = []
    seen = set()
    for path in files:
        try:
            head = next(csv.reader(open(path)), None)
        except Exception:
            continue
        if not head or "kernel" not in head or len(head) < 38:
            continue
        for column in head:
            if column not in seen:
                seen.add(column)
                columns.append(column)

    rows = 0
    skipped = []          # (file, why) -- reported, never swallowed
    captures = set()
    created = False

    for path in files:
        capture = os.path.basename(os.path.dirname(path))
        try:
            data = list(csv.DictReader(open(path)))
        except Exception as exc:
            skipped.append((path, f"unreadable: {exc.__class__.__name__}"))
            continue
        if not data:
            skipped.append((path, "empty"))
            continue
        if "kernel" not in data[0] or len(data[0]) < 38:
            skipped.append((path, f"not the sweep schema ({len(data[0])} columns)"))
            continue

        if not created:
            extra = ["capture", "instance", "part", "uarch", "source_file"]
            db.execute("CREATE TABLE m (%s)"
                       % ",".join('"%s"' % c for c in columns + extra))
            created = True

        instance, part, uarch = identify(capture)
        captures.add(capture)
        for row in data:
            values = [coerce(row.get(c), c) for c in columns]
            values += [capture, instance, part, uarch, os.path.basename(path)]
            db.execute("INSERT INTO m VALUES (%s)" % ",".join("?" * len(values)),
                       values)
            rows += 1

    if not created:
        sys.exit("ingest: no file matched the sweep schema; nothing to build")

    for index in ("kernel", "algorithm", "isa", "capture", "part", "compiler",
                  "threads", "status"):
        db.execute('CREATE INDEX i_%s ON m("%s")' % (index, index))
    db.commit()

    print("  %d rows from %d files, %d captures -> %s (%.0f KB)"
          % (rows, len(files) - len(skipped), len(captures), db_path,
             os.path.getsize(db_path) / 1024))

    unknown = sorted({identify(c)[0] for c in captures if identify(c)[1] is None})
    if unknown:
        print("  unrecognised instance prefixes (add them to PARTS): %s"
              % ", ".join(unknown))

    # Skipped files are printed, not counted away. Most are perf output or
    # ad-hoc formats and are meant to be skipped -- but a capture script that
    # quietly stopped emitting the sweep schema looks exactly the same, and
    # that has happened once already.
    if skipped:
        by_reason = {}
        for path, why in skipped:
            by_reason.setdefault(why, []).append(os.path.relpath(path, root))
        print("  skipped %d files:" % len(skipped))
        for why, paths in sorted(by_reason.items()):
            print("    %-42s %d  e.g. %s" % (why, len(paths), paths[0]))


if __name__ == "__main__":
    main()
