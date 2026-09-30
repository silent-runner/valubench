#!/usr/bin/env python3
"""
compare.py -- diff two sets of valubench results.

SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026, The valubench authors. See LICENSE.

One run prints one number; the question that actually matters is whether it
moved. This reads two sets of results -- single JSON files, directories of
them, or the CSV that sweep.py writes -- pairs up the points that measured the
same thing, and reports the change.

Two properties of the benchmark do the heavy lifting here:

  * The workload id ("md5-full-55x1") changes whenever the work per hash
    changes, so points from different workloads are never silently compared.
    docs/research.md 1.1.

  * The verification checksum is invariant to lanes, streams, threads and
    devices, so two runs of the same workload must produce the same value on
    any machine and any kernel. A mismatch is not a slow result, it is a wrong
    one, and it is reported as a hard failure rather than a delta.

A delta is only called a change when it clears both bars: the 10% this project
calls significant, and the run-to-run noise both sides reported. The second
matters more than it looks -- all-core figures on a thermally limited part
carry 20-30% CoV, which is wider than most of the differences anyone wants to
read into them.

  ./tools/compare.py before.json after.json
  ./tools/compare.py results-old/ results-new/
  ./tools/compare.py base.csv new.csv --threshold 5

Exit status: 0 no significant regression, 1 at least one regression,
             2 usage error, 3 results not comparable (checksum mismatch).
"""

import argparse
import csv
import json
import math
import os
import sys

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_INCOMPARABLE = 3

# The project's bar: improvements of 10% or more are significant.
DEFAULT_THRESHOLD = 10.0

# What identifies a measurement. Two records with the same key measured the
# same thing and may be compared; anything else is a different question.
KEY_FIELDS = ("workload", "kernel", "threads", "transfer_mode",
              "working_set_bytes", "variant")


def device_variant(compile_mode="", compiler="", primitives="", host_memory="",
                   geometry_source="", global_work="", local_work=""):
    """
    What else distinguishes two runs of one device kernel: how it was compiled
    (and whether the code was imported), whether its primitives were steered,
    what a streaming upload read, and a pinned launch. Two CUDA rows of one
    sweep can differ only here, and pairing them by kernel name alone kept the
    first and dropped the other. A tuned launch is left out of the key on
    purpose -- the tuner can choose differently between identical runs, and
    that is a result to compare, not a reason not to. Empty for CPU rows and
    for files from before these existed, which therefore still pair.
    """
    parts = []
    if compiler == "imported":
        parts.append("imported")
    elif compile_mode and compile_mode != "driver":
        parts.append(compile_mode)
    if primitives == "neutral":
        parts.append("neutral")
    if host_memory:
        parts.append(host_memory)
    if geometry_source == "pinned":
        parts.append("%sx%s" % (global_work, local_work))
    return "/".join(parts)


class Record:
    """One measured point, from whichever file format it arrived in."""

    def __init__(self, workload, kernel, threads, transfer_mode,
                 working_set_bytes, hashes_per_sec, cov_percent, checksum,
                 verified, source, cpu="", version="", variant=""):
        self.workload = workload
        self.kernel = kernel
        self.threads = int(threads)
        self.transfer_mode = transfer_mode or ""
        self.working_set_bytes = int(working_set_bytes)
        self.hashes_per_sec = float(hashes_per_sec)
        self.cov_percent = float(cov_percent)
        self.checksum = checksum
        self.verified = verified
        self.source = source
        self.cpu = cpu
        self.version = version
        self.variant = variant

    @property
    def key(self):
        return (self.workload, self.kernel, self.threads, self.transfer_mode,
                self.working_set_bytes, self.variant)

    def label(self):
        parts = [self.workload, self.kernel]
        if self.variant:
            parts.append(self.variant)
        if self.threads != 1:
            parts.append("%dt" % self.threads)
        else:
            parts.append("1t")
        if self.transfer_mode:
            parts.append(self.transfer_mode)
        return " ".join(parts)


def record_from_json(doc, source):
    p, r, b, k = (doc["parameters"], doc["result"], doc["benchmark"],
                  doc["kernel"])
    dev = doc.get("device", {})
    env = doc.get("environment", {})
    return Record(
        workload=b["workload"],
        kernel=k["name"],
        threads=p["threads"],
        transfer_mode=dev.get("transfer_mode", ""),
        working_set_bytes=p["working_set_bytes"],
        hashes_per_sec=r["median"],
        cov_percent=r["cov_percent"],
        checksum=doc["verification"]["checksum"],
        verified=bool(doc["verification"]["verified"]),
        source=source,
        cpu=env.get("cpu", ""),
        version=b.get("version", ""),
        variant=device_variant(
            dev.get("compile_mode", ""), dev.get("compiler", ""),
            dev.get("primitives", ""), dev.get("host_memory", ""),
            dev.get("geometry_source", ""), dev.get("global_work", ""),
            dev.get("local_work", "")),
    )


def record_from_csv_row(row, source):
    return Record(
        workload=row["workload"],
        kernel=row["kernel"],
        threads=row["threads"],
        transfer_mode=row.get("transfer_mode", ""),
        working_set_bytes=row["working_set_bytes"],
        hashes_per_sec=row["hashes_per_sec"],
        cov_percent=row["cov_percent"],
        checksum=row["checksum"],
        verified=row.get("verified", "true") == "true",
        source=source,
        cpu=row.get("cpu", ""),
        version=row.get("valubench_version", ""),
        variant=device_variant(
            row.get("compile_mode", ""), row.get("device_compiler", ""),
            row.get("primitives", ""), row.get("host_memory", ""),
            row.get("geometry_source", ""), row.get("global_work", ""),
            row.get("local_work", "")),
    )


def usage_error(msg):
    """Input this cannot compare: exit 2, as the docstring promises. A bare
    sys.exit(msg) exits 1, which is this tool's word for a regression."""
    print("compare: %s" % msg, file=sys.stderr)
    sys.exit(EXIT_USAGE)


def load_path(path):
    """
    Read one result file, or every result file in a directory.

    A directory is read recursively, because that is the shape capture.sh
    leaves behind: several sweep CSVs and some environment capture in one
    place. Files that are not results are skipped rather than fatal -- a
    results directory legitimately contains build logs and clinfo output.
    """
    records, unreadable = [], []

    if os.path.isdir(path):
        names = []
        for dirpath, _dirnames, filenames in os.walk(path):
            for fn in sorted(filenames):
                if fn.endswith((".json", ".csv")):
                    names.append(os.path.join(dirpath, fn))
        if not names:
            usage_error("no .json or .csv files under %s" % path)
    else:
        names = [path]

    for name in names:
        try:
            got = load_file(name)
        except (OSError, ValueError, KeyError) as e:
            unreadable.append((name, e))
            continue
        records.extend(got)

    if not records:
        detail = "".join("\n  %s: %s" % (n, e) for n, e in unreadable)
        usage_error("no usable results in %s%s" % (path, detail))

    for name, e in unreadable:
        print("compare: skipping %s (%s)" % (name, e), file=sys.stderr)

    return records


def load_file(path):
    label = os.path.basename(path)

    if path.endswith(".json"):
        with open(path) as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError("not a valubench result (a JSON %s)" % type(doc).__name__)
        schema = str(doc.get("schema", ""))
        if not schema.startswith("valubench/result/"):
            raise ValueError("not a valubench result (schema %r)" % schema)
        return [record_from_json(doc, label)]

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty CSV")
    missing = [c for c in ("workload", "kernel", "hashes_per_sec", "checksum")
               if c not in rows[0]]
    if missing:
        raise ValueError("not a sweep CSV (no %s column)" % ", ".join(missing))

    out = []
    for row in rows:
        # sweep.py keeps rows it could not measure out of the CSV entirely, but
        # a hand-edited file may have blanks. Skip rather than crash.
        if not row.get("hashes_per_sec"):
            continue
        out.append(record_from_csv_row(row, label))
    return out


def index(records, side):
    """Key -> record, complaining about duplicates rather than picking one."""
    by_key = {}
    for rec in records:
        if rec.key in by_key:
            print("compare: %s has %s twice (%s and %s); keeping the first"
                  % (side, rec.label(), by_key[rec.key].source, rec.source),
                  file=sys.stderr)
            continue
        by_key[rec.key] = rec
    return by_key


def verdict(delta_pct, noise_pct, threshold):
    """
    Classify a change.

    Two bars, and a change has to clear both. The threshold is the project's
    own definition of significant (10%). The noise floor is what the
    two runs said about themselves: quadrature of the two CoVs, which is the
    right combination for independent variation and is roughly the wider of the
    two when one dominates.

    Below the noise floor the sign of the delta means nothing at all, which is
    a different statement from "small but real", so they are named differently.
    """
    if abs(delta_pct) < noise_pct:
        return "noise"
    if abs(delta_pct) < threshold:
        return "minor"
    return "FASTER" if delta_pct > 0 else "SLOWER"


def compare(base, new, threshold):
    base_idx = index(base, "base")
    new_idx = index(new, "new")

    matched, mismatched, only_base, only_new = [], [], [], []

    for key in sorted(base_idx.keys() & new_idx.keys()):
        b, n = base_idx[key], new_idx[key]

        # Same workload, same messages, so the digests XOR to the same value on
        # any machine. If they do not, one side computed something else and no
        # throughput comparison between them means anything.
        if b.checksum != n.checksum:
            mismatched.append((b, n))
            continue

        delta = (n.hashes_per_sec - b.hashes_per_sec) / b.hashes_per_sec * 100.0
        noise = math.sqrt(b.cov_percent ** 2 + n.cov_percent ** 2)
        matched.append((b, n, delta, noise, verdict(delta, noise, threshold)))

    for key in sorted(base_idx.keys() - new_idx.keys()):
        only_base.append(base_idx[key])
    for key in sorted(new_idx.keys() - base_idx.keys()):
        only_new.append(new_idx[key])

    return matched, mismatched, only_base, only_new


def human(matched, mismatched, only_base, only_new, threshold, f):
    if mismatched:
        print("CHECKSUM MISMATCH -- these did not compute the same answer\n",
              file=f)
        for b, n in mismatched:
            print("  %s" % b.label(), file=f)
            print("    base %s  (%s)" % (b.checksum, b.source), file=f)
            print("    new  %s  (%s)" % (n.checksum, n.source), file=f)
        print("", file=f)

    if matched:
        width = max(len(b.label()) for b, _n, _d, _s, _v in matched)
        width = max(width, 12)
        print("%-*s  %12s  %12s  %8s  %7s  %s"
              % (width, "point", "base MH/s", "new MH/s", "delta", "noise",
                 "verdict"), file=f)
        print("-" * (width + 56), file=f)
        for b, n, delta, noise, v in sorted(matched, key=lambda m: m[2]):
            print("%-*s  %12.2f  %12.2f  %+7.1f%%  %6.1f%%  %s"
                  % (width, b.label(), b.hashes_per_sec / 1e6,
                     n.hashes_per_sec / 1e6, delta, noise, v), file=f)
        print("", file=f)

    faster = [m for m in matched if m[4] == "FASTER"]
    slower = [m for m in matched if m[4] == "SLOWER"]
    noisy = [m for m in matched if m[4] == "noise"]

    print("%d point%s compared at a %.3g%% threshold: %d faster, %d slower, "
          "%d within noise"
          % (len(matched), "" if len(matched) == 1 else "s", threshold,
             len(faster), len(slower), len(noisy)), file=f)

    # A comparison where most points cannot resolve the threshold is not a
    # result, it is a measurement problem. Say so rather than letting the table
    # imply more than it can carry.
    inconclusive = [m for m in matched if m[3] > threshold]
    if inconclusive:
        print("  %d point%s reported more run-to-run noise than the threshold "
              "itself -- those rows cannot resolve a %.3g%% change at all"
              % (len(inconclusive), "" if len(inconclusive) == 1 else "s",
                 threshold), file=f)

    if only_base:
        print("  %d point%s only in base (%s%s)"
              % (len(only_base), "" if len(only_base) == 1 else "s",
                 ", ".join(r.label() for r in only_base[:3]),
                 ", ..." if len(only_base) > 3 else ""), file=f)
    if only_new:
        print("  %d point%s only in new (%s%s)"
              % (len(only_new), "" if len(only_new) == 1 else "s",
                 ", ".join(r.label() for r in only_new[:3]),
                 ", ..." if len(only_new) > 3 else ""), file=f)


def as_json(matched, mismatched, only_base, only_new, threshold, f):
    doc = {
        "schema": "valubench/comparison/1",
        "threshold_percent": threshold,
        "points": [
            {
                "workload": b.workload,
                "kernel": b.kernel,
                "threads": b.threads,
                "transfer_mode": b.transfer_mode,
                "working_set_bytes": b.working_set_bytes,
                "base_hashes_per_sec": b.hashes_per_sec,
                "new_hashes_per_sec": n.hashes_per_sec,
                "delta_percent": delta,
                "noise_percent": noise,
                "verdict": v,
            }
            for b, n, delta, noise, v in matched
        ],
        "checksum_mismatches": [
            {
                "workload": b.workload,
                "kernel": b.kernel,
                "base_checksum": b.checksum,
                "new_checksum": n.checksum,
            }
            for b, n in mismatched
        ],
        "only_in_base": [r.label() for r in only_base],
        "only_in_new": [r.label() for r in only_new],
    }
    json.dump(doc, f, indent=2)
    f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description="Compare two sets of valubench results.",
        epilog="Each argument is a result JSON, a sweep CSV, or a directory "
               "of either.")
    ap.add_argument("base", help="the results to compare against")
    ap.add_argument("new", help="the results to judge")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    metavar="PCT",
                    help="percent change called significant (default %.3g, "
                         "the significance bar)" % DEFAULT_THRESHOLD)
    ap.add_argument("--json", action="store_true",
                    help="emit the comparison as JSON")
    args = ap.parse_args()

    if args.threshold < 0:
        usage_error("--threshold must be >= 0")

    for path in (args.base, args.new):
        if not os.path.exists(path):
            usage_error("no such file or directory: %s" % path)

    base = load_path(args.base)
    new = load_path(args.new)

    matched, mismatched, only_base, only_new = compare(base, new,
                                                       args.threshold)

    if not matched and not mismatched:
        print("compare: nothing in common between %s and %s.\n"
              "  Points are paired by workload, kernel, threads, transfer mode, "
              "working set and\n  device variant (compile mode, primitives, "
              "host memory, pinned launch);\n  a difference in any of those is "
              "a different measurement, not a slower one." % (args.base, args.new),
              file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        as_json(matched, mismatched, only_base, only_new, args.threshold,
                sys.stdout)
    else:
        human(matched, mismatched, only_base, only_new, args.threshold,
              sys.stdout)

    if mismatched:
        return EXIT_INCOMPARABLE
    if any(m[4] == "SLOWER" for m in matched):
        return EXIT_REGRESSION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
