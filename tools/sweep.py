#!/usr/bin/env python3
"""
sweep.py -- drive valubench across a parameter grid and collect the results.

SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026, The valubench authors. See LICENSE.

Python's role in this project is orchestration only. It never enters a timed
region: it invokes the C binary, parses the JSON the binary emits, and arranges
the results. The benchmark stays fully usable without it.

Standard library only -- no numpy, no pandas. valubench is expected to run on a
bare Linux box with no network access, so anything outside the base install is a
liability rather than a convenience.

Emits one CSV row per measured point, in the shape mixbench uses, so the output
feeds straight into whatever plots it.

  ./tools/sweep.py --message-bytes 55,1015,4087 > sweep.csv

  ./tools/sweep.py --kernel avx2-s4 --threads 1 \\
      --message-bytes 64:4096:*2 --iterations 1,4,16 --csv surface.csv

  # the goal 2 sweep: where does compute overtake the PCIe link?
  ./tools/sweep.py --algorithm md5,sha1,sha512 --transfer stream \\
      --working-set-kb 262144 --iterations 1:256:*2 --csv crossover.csv

  # and the purchasing question: when does offloading beat the whole CPU?
  ./tools/sweep.py --algorithm md5 --transfer stream --where device,cpu \\
      --message-bytes 64 --working-set-kb 262144 --iterations 1:64:*2
"""

import argparse
import csv
import itertools
import math
import json
import os
import shutil
import subprocess
import sys
import time

# What this script knows about valubench, it asks valubench for.
#
# `valubench --list --json` reports the algorithms and their geometry, every
# kernel and whether it runs here, the parameter limits, the defaults and the
# exit codes. All of that used to be transcribed into this file, and the
# transcription drifted: the per-algorithm minimum message length was carried
# as one constant, which silently skipped or failed every SHA-512 point with
# iterations > 1. Facts the binary knows come from the binary.


class Capabilities:
    """What the binary reports about itself, on this machine."""

    def __init__(self, doc):
        self.version = doc["benchmark"]["version"]
        self.algorithms = tuple(a["name"] for a in doc["algorithms"])
        self.transfer_modes = tuple(doc["transfer_modes"])
        # Autotune restrictions. Older binaries have no such notion, so this
        # degrades to "any" rather than failing.
        self.where_filters = tuple(doc.get("where_filters", ("any",)))
        self.limits = doc["limits"]
        self.exit = doc["exit_codes"]
        self.kernels = {k["name"]: k for k in doc["kernels"]}

        # Iterated hashing feeds the digest back over the head of the message,
        # so a message shorter than a digest has nowhere to put it. Per
        # algorithm: 16, 20 and 64 bytes for md5, sha1 and sha512.
        self.min_iter_bytes = {a["name"]: a["min_iteration_message_bytes"]
                               for a in doc["algorithms"]}

    @classmethod
    def query(cls, binary):
        try:
            proc = subprocess.run([binary, "--list", "--json"],
                                  capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            sys.exit("sweep: cannot query %s: %s" % (binary, e))

        if proc.returncode != 0 or not proc.stdout.strip():
            sys.exit("sweep: '%s --list --json' produced nothing. A binary\n"
                     "       older than the capability dump cannot be driven "
                     "by this script -- run 'make'." % binary)
        try:
            doc = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            sys.exit("sweep: '%s --list --json' is not JSON: %s" % (binary, e))

        schema = str(doc.get("schema", ""))
        if not schema.startswith("valubench/capabilities/"):
            sys.exit("sweep: unexpected schema %r from '%s --list --json'"
                     % (schema, binary))
        return cls(doc)

    def resolve_kernel(self, name, alg):
        """
        Qualify a kernel name for one algorithm.

        Registry names are '<alg>/<isa>-s<streams>'. A bare '<isa>-s<streams>'
        means "that kernel, for whichever algorithm this point measures", which
        is what sweeping one ISA across algorithms wants. Returns the full name,
        or None if no such kernel is registered.
        """
        full = name if "/" in name else "%s/%s" % (alg, name)
        return full if full in self.kernels else None

    def available(self, full_name):
        return bool(self.kernels[full_name]["available"])

CSV_COLUMNS = [
    # parameters
    "workload",
    "algorithm",
    "message_bytes",
    "blocks_per_message",
    "iterations",
    "threads",
    "point_id",
    "working_set_kb",
    "working_set_bytes",
    "batch_messages",
    "transfer_mode",
    # what ran
    "kernel",
    "isa",
    "lanes",
    "streams",
    "runs_on",
    "device",
    # results
    "hashes_per_sec",
    "hashes_per_sec_min",
    "compressions_per_sec",
    "message_bytes_per_sec",
    "cov_percent",
    "stable",
    # device only; blank for CPU rows
    "kernel_busy_pct",
    "transfer_busy_pct",
    "transfer_gbytes_per_sec",
    "compute_transfer_ratio",
    "bound_by",
    "kernel_ns_per_pass",
    "transfer_ns_per_pass",
    # provenance
    "verified",
    "checksum",
    "samples",
    "cpu",
    "virtualized",
    "smt_active",
    "pinned_cpus",
    "can_pin",
    "governor",
    "governor_at_end",
    "freq_khz_at_end",
    "temp_milli_c",
    "temp_milli_c_at_end",
    "loadavg_1min",
    "compiler",
    "valubench_version",
    "status",
]


def parse_list(spec, what):
    """
    Parse an axis specification into a list of ints.

        "55"              -> [55]
        "55,1015,4087"    -> [55, 1015, 4087]
        "64:4096:*2"      -> [64, 128, 256, ..., 4096]     geometric
        "1:10:+3"         -> [1, 4, 7, 10]                 arithmetic

    Ranges are inclusive of the start and never exceed the stop.
    """
    values = []

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            parts = token.split(":")
            if len(parts) != 3:
                raise ValueError(
                    "%s: range needs start:stop:step, got %r" % (what, token))
            try:
                start, stop = int(parts[0]), int(parts[1])
            except ValueError:
                raise ValueError("%s: bad range bounds in %r" % (what, token))

            step = parts[2].strip()
            if not step or step[0] not in "*+":
                raise ValueError(
                    "%s: step must start with '*' or '+', got %r" % (what, step))
            try:
                amount = int(step[1:])
            except ValueError:
                raise ValueError("%s: bad step in %r" % (what, token))

            if start < 1 or stop < start:
                raise ValueError("%s: empty range %r" % (what, token))
            if step[0] == "*" and amount < 2:
                raise ValueError("%s: geometric step must be >= 2" % what)
            if step[0] == "+" and amount < 1:
                raise ValueError("%s: arithmetic step must be >= 1" % what)

            v = start
            while v <= stop:
                values.append(v)
                v = v * amount if step[0] == "*" else v + amount
        else:
            try:
                values.append(int(token))
            except ValueError:
                raise ValueError("%s: not an integer: %r" % (what, token))

    if not values:
        raise ValueError("%s: no values" % what)

    # Preserve order, drop duplicates.
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_choice_list(spec, what, allowed):
    """Parse a comma-separated axis of names, validated against `allowed`."""
    out, seen = [], set()

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if allowed and token not in allowed:
            raise ValueError("%s: unknown value %r (choose from %s)"
                             % (what, token, ", ".join(allowed)))
        if token not in seen:
            seen.add(token)
            out.append(token)

    if not out:
        raise ValueError("%s: no values" % what)
    return out


def build_grid(args, caps):
    """
    Cartesian product of the axes, with invalid combinations dropped.

    Flat rather than nested: eight axes as a pyramid of for-loops put the body
    twenty columns in and made adding the ninth a reindentation exercise.
    """
    points, skipped = [], []

    axes = itertools.product(args.algorithm, args.kernel or [None], args.where,
                             args.transfer, args.message_bytes,
                             args.iterations, args.working_set_kb,
                             args.threads)

    for alg, kern, where, xfer, mb, it, ws, th in axes:
        # A forced kernel only computes one algorithm; pairing it with the
        # others would be a guaranteed usage error on every such point.
        if kern and "/" in kern and kern.split("/")[0] != alg:
            continue

        # Resolve against the registry the binary reports rather than trusting
        # the spelling. A name that does not exist, or an ISA this machine
        # cannot run, is one line of output instead of a failure on every point
        # that used it.
        resolved = None
        if kern:
            resolved = caps.resolve_kernel(kern, alg)
            if resolved is None:
                skipped.append("%s kernel=%s: no such kernel" % (alg, kern))
                continue
            if not caps.available(resolved):
                skipped.append("%s: this machine cannot run %s"
                               % (resolved, caps.kernels[resolved]["isa"]))
                continue

            # A forced kernel already decides where it runs, so a filter that
            # excludes it is a contradiction the binary would reject. Drop it
            # quietly -- it is a grid artefact, not a request anyone made.
            if where != "any" and caps.kernels[resolved]["where"] != where:
                continue

        # Iterated hashing feeds the digest back over the head of the message,
        # so a message shorter than a digest has nowhere to put it.
        if it > 1 and mb < caps.min_iter_bytes[alg]:
            skipped.append("%s message-bytes=%d iterations=%d: iterated "
                           "hashing needs >= %d byte messages"
                           % (alg, mb, it, caps.min_iter_bytes[alg]))
            continue

        points.append({
            "algorithm": alg,
            "kernel": resolved,
            "where": where,
            "transfer": xfer,
            "message_bytes": mb,
            "iterations": it,
            "working_set_kb": ws,
            "threads": th,
        })

    # One reason is one line, however many grid points it eliminated.
    seen, unique = set(), []
    for note in skipped:
        if note not in seen:
            seen.add(note)
            unique.append(note)

    return points, unique


def point_id(p):
    """
    A stable identifier for a grid point, written into every row.

    Only what the command line asked for -- never what the machine decided.
    Matching a request against a result does not work: `transfer_mode` comes
    back empty for CPU kernels whatever was requested, and `kernel` comes back
    as autotune's choice where the request was "pick one". Recording the request
    verbatim sidesteps both, and reads well enough in the CSV to be useful to a
    person scanning it.
    """
    return ("alg=%s;kernel=%s;where=%s;transfer=%s;mb=%s;it=%s;ws=%s;thr=%s"
            % (p["algorithm"], p["kernel"] or "auto", p.get("where", "any"),
               p["transfer"], p["message_bytes"], p["iterations"],
               p["working_set_kb"], p["threads"]))


def load_completed(path):
    """
    Keys already present in an existing CSV, and whether its header matches.

    A grid that dies partway through -- a dropped connection, a preempted
    instance, an hour of rented time running out -- should not have to start
    over. Returns (keys, header_ok); an unreadable or absent file is simply no
    completed points.
    """
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
            f.seek(0)
            header = next(csv.reader(f), None)
    except (OSError, StopIteration):
        return set(), True
    if header is None:
        return set(), True
    if header != CSV_COLUMNS:
        return set(), False
    return {r["point_id"] for r in rows
            if r.get("hashes_per_sec") and r.get("point_id")}, True


def precompute_references(args, points):
    """Compute every point's expected checksum up front, one pass per corpus.

    The iteration scheme is a chain -- hash, write the digest over the head of
    the message, hash again -- so the digest after k iterations is a *prefix* of
    the chain for any larger k. A ladder of 1, 2, 4 ... 1024 therefore asks for
    the same walk over and over, once per point, and each walk is
    messages x iterations of scalar hashing before anything is measured. On an
    A10 that reached 733 seconds for a single point.

    Walking each message once to the largest count and snapshotting at every
    requested one replaces the whole ladder, and the binary splits that walk
    across the cores. Points at one iteration are left alone: the reference
    there is a single hash per message and costs nothing to redo.

    Returns {(algorithm, message_bytes, working_set_kb): {iterations: hex}}.
    A key missing from the result simply means those points verify the old way,
    so a failure here costs time and never correctness.
    """
    ladders = {}
    for p in points:
        if p["iterations"] < 2:
            continue
        key = (p["algorithm"], p["message_bytes"], p["working_set_kb"])
        ladders.setdefault(key, set()).add(p["iterations"])

    out = {}
    for key, iters in sorted(ladders.items()):
        alg, mb, ws = key
        rungs = sorted(iters)
        cmd = [args.bin, "--reference-ladder", ",".join(str(i) for i in rungs),
               "--algorithm", alg, "--message-bytes", str(mb),
               "--working-set-kb", str(ws)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout)
            doc = json.loads(proc.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue
        if proc.returncode != 0:
            continue
        out[key] = {c["iterations"]: c["checksum"] for c in doc["checksums"]}

    return out


def run_point(args, caps, point, refs=None):
    """Run one configuration. Returns (status, result_dict_or_None)."""
    cmd = [
        args.bin, "--json",
        "--algorithm", point["algorithm"],
        "--transfer", point["transfer"],
        "--message-bytes", str(point["message_bytes"]),
        "--iterations", str(point["iterations"]),
        "--working-set-kb", str(point["working_set_kb"]),
        "--threads", str(point["threads"]),
        "--samples", str(args.samples),
        "--time-ms", str(args.time_ms),
        "--warmup-ms", str(args.warmup_ms),
    ]
    if refs:
        got = refs.get((point["algorithm"], point["message_bytes"],
                        point["working_set_kb"]), {}).get(point["iterations"])
        if got:
            cmd += ["--expect", got]
    if point["kernel"]:
        cmd += ["--kernel", point["kernel"]]
    elif point.get("where", "any") != "any":
        cmd += ["--where", point["where"]]
    if not args.pin:
        cmd += ["--no-pin"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None
    except OSError as e:
        return "error: %s" % e, None

    # Exit 3 means the result is too noisy to trust, but it is still a real
    # measurement with valid JSON -- keep it and let the row say so.
    if proc.returncode in (caps.exit["ok"], caps.exit["noisy"]):
        try:
            return ("ok" if proc.returncode == caps.exit["ok"] else "noisy",
                    json.loads(proc.stdout))
        except json.JSONDecodeError:
            return "bad-json", None

    if proc.returncode == caps.exit["verify_failed"]:
        return "VERIFICATION FAILED", None
    if proc.returncode == caps.exit["usage"]:
        first = (proc.stderr or "").strip().splitlines()
        return "usage error: %s" % (first[0] if first else "?"), None

    return "exit %d" % proc.returncode, None


def pct(v):
    return "" if v is None else "%.1f" % (v * 100.0)


def row_from_result(d, status, point=None):
    """A CSV row from a result document, or a row saying it could not be read.

    Never raises. One malformed document used to throw KeyError out of the grid
    loop and stop the sweep: rows already written survived, but the
    balance-point and break-even summaries -- the reason to run a grid -- did
    not. A point that cannot be parsed is data too, and the run continues.
    """
    try:
        return _row_from_result(d, status, point)
    except Exception as exc:                       # noqa: BLE001 - see above
        row = dict.fromkeys(CSV_COLUMNS, "")
        if point:
            row.update({k: v for k, v in point.items() if k in CSV_COLUMNS})
            row["point_id"] = point_id(point)
        row["status"] = "unreadable-result (%s)" % exc.__class__.__name__
        return row


def _row_from_result(d, status, point=None):
    # .get throughout: one malformed result used to raise KeyError out of the
    # grid loop and stop the sweep. Rows already written survived, but the
    # balance-point and break-even summaries -- the reason to run a grid at all
    # -- did not. A row that cannot be parsed is worth recording as such and
    # stepping over.
    p, r = d.get("parameters", {}), d.get("result", {})
    k = d.get("kernel", {})
    e = d.get("environment", {})
    b = d.get("benchmark", {})

    # Absent entirely for CPU kernels, so every device field is optional.
    dev = d.get("device", {})

    return {
        "workload": b["workload"],
        "algorithm": b["algorithm"],
        "message_bytes": p["message_bytes"],
        "blocks_per_message": p["blocks_per_message"],
        "iterations": p["iterations"],
        "threads": p["threads"],
        "point_id": point_id(point) if point else "",
        "working_set_kb": p["working_set_bytes"] // 1024,
        "working_set_bytes": p["working_set_bytes"],
        "batch_messages": p["batch_messages"],
        "transfer_mode": dev.get("transfer_mode", ""),
        "kernel": k["name"],
        "isa": k["isa"],
        "lanes": k["lanes"],
        "streams": k["streams"],
        "runs_on": k["runs_on"],
        "device": dev.get("name", ""),
        "hashes_per_sec": "%.6g" % r["median"],
        "hashes_per_sec_min": "%.6g" % r["min"],
        "compressions_per_sec": "%.6g" % r["compressions_per_second"],
        "message_bytes_per_sec": "%.6g" % r["message_bytes_per_second"],
        "cov_percent": "%.4g" % r["cov_percent"],
        "stable": "true" if r["stable"] else "false",
        "kernel_busy_pct": pct(dev.get("kernel_busy_fraction")),
        "transfer_busy_pct": pct(dev.get("transfer_busy_fraction")),
        "transfer_gbytes_per_sec":
            "" if "transfer_gbytes_per_sec" not in dev
            else "%.4g" % dev["transfer_gbytes_per_sec"],
        "compute_transfer_ratio":
            "" if "compute_transfer_ratio" not in dev
            else "%.4g" % dev["compute_transfer_ratio"],
        "bound_by": dev.get("bound_by", ""),
        "kernel_ns_per_pass": dev.get("kernel_ns_per_pass", ""),
        "transfer_ns_per_pass": dev.get("transfer_ns_per_pass", ""),
        "verified": "true" if d["verification"]["verified"] else "false",
        "checksum": d["verification"]["checksum"],
        "samples": len(r["samples"]),
        "cpu": e["cpu"],
        # yes/no/unknown. Every ARM machine measured so far is a VM
        # that no x86-style check could identify as one.
        "virtualized": e.get("virtualized", ""),
        # SMT reached the JSON but never the CSV, so it never reached any
        # cross-machine comparison -- and it is one of the larger sources of
        # run-to-run variance.
        "smt_active": e.get("smt_active", ""),
        # Neither did this, which is worse: pinned_cpus exists because a pool
        # that collapsed onto one core under-reported by 3.4x while reporting
        # verified, and the results README tells readers to check it against
        # threads_used. They could not -- it stopped at the JSON.
        "pinned_cpus": e.get("pinned_cpus", ""),
        # And pinned_cpus is 0 both when a pin was refused and when the
        # platform had no affinity API to attempt, so it needs can_pin beside
        # it or a query cannot tell a broken pool from a Mac.
        "can_pin": e.get("can_pin", ""),
        "governor": e["governor"],
        # Sampled again after the timed region: a clock or temperature read
        # only at startup describes a machine that has not run yet.
        "governor_at_end": e.get("governor_at_end", ""),
        "freq_khz_at_end": e.get("freq_khz_at_end", ""),
        "temp_milli_c": e.get("temp_milli_c", ""),
        "temp_milli_c_at_end": e.get("temp_milli_c_at_end", ""),
        "loadavg_1min": e.get("loadavg_1min"),
        "compiler": e["compiler"],
        "valubench_version": b["version"],
        "status": status,
    }


def balance_point(rows):
    """
    Solve for the iteration count at which compute exactly balances transfer.

    The workload is defined so that every iteration is identical work, and the
    same bytes cross the link at every point regardless of iteration count. So
    over a streaming sweep:

        kernel_ns(N) = a + b*N        transfer_ns = T   (constant in N)

    and the balance point is the N where those are equal:

        N* = (T - a) / b

    This *solves* for N* rather than bracketing it, which matters because a
    geometric sweep only ever brackets to within its own step -- a crossing
    between 2 and 4 is a factor-of-two answer. Separating the intercept `a`
    also removes the drift in the cruder estimate N/ratio, which silently
    charges the fixed per-launch cost to compute.

    Returns a dict, or None if the sweep cannot support the fit.
    """
    pts = []
    for r in rows:
        try:
            n = float(r["iterations"])
            k = float(r["kernel_ns_per_pass"])
            t = float(r["transfer_ns_per_pass"])
        except (ValueError, KeyError, TypeError):
            continue
        if n > 0 and k > 0 and t > 0:
            pts.append((n, k, t))

    if len(pts) < 2:
        return None

    n_mean = sum(p[0] for p in pts) / len(pts)
    k_mean = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - n_mean) ** 2 for p in pts)
    if sxx == 0:
        return None                      # every point at the same iteration count
    sxy = sum((p[0] - n_mean) * (p[1] - k_mean) for p in pts)

    b = sxy / sxx                        # ns of compute per iteration
    a = k_mean - b * n_mean              # fixed per-launch ns
    if b <= 0:
        return None

    # How linear was it? A poor fit means the model does not hold here and the
    # answer should not be quoted -- e.g. the corpus fell out of device cache
    # partway up the sweep, or the launch was too short to measure.
    ss_tot = sum((p[1] - k_mean) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Transfer should be flat across the sweep; its spread is the honesty check
    # on the whole measurement, since it is the same bytes every point.
    ts = [p[2] for p in pts]
    t_mean = sum(ts) / len(ts)
    t_spread = (max(ts) - min(ts)) / t_mean if t_mean > 0 else 0.0

    n_star = (t_mean - a) / b

    return {
        "n_star": n_star,
        "per_iter_ns": b,
        "launch_overhead_ns": a,
        "transfer_ns": t_mean,
        "transfer_spread": t_spread,
        "r2": r2,
        "points": len(pts),
    }


def report_balance(rows, out):
    """Print the balance point per (algorithm, kernel) group."""
    groups = {}
    for r in rows:
        if r.get("transfer_mode") != "stream":
            continue
        groups.setdefault((r["algorithm"], r["kernel"]), []).append(r)

    printed = False
    for (alg, kern), rs in groups.items():
        fit = balance_point(rs)
        if not fit:
            continue
        if not printed:
            print("\nPCIe balance point  (compute == transfer)", file=out)
            printed = True

        n = fit["n_star"]
        warn = ""
        # A two-point fit has r2 = 1.0000 by construction, so the honesty
        # gate below passed most confidently exactly where it was least
        # earned. Three points is the minimum at which r2 says anything.
        if fit["points"] < 3:
            warn += ("   [%d points -- r2 is 1.0 by construction, not a fit]"
                     % fit["points"])
        elif fit["r2"] < 0.98:
            warn += "   [nonlinear, r2=%.3f -- do not quote]" % fit["r2"]
        if fit["transfer_spread"] > 0.15:
            warn += "   [transfer varied %.0f%% across the sweep]" % (
                fit["transfer_spread"] * 100)

        print("  %-16s N* = %.2f iterations%s" % (kern, n, warn), file=out)
        print("      compute-bound from %d iterations up" % max(1, math.ceil(n)),
              file=out)
        print("      %.3f ms per iteration, %.3f ms fixed launch cost, "
              "%.3f ms transfer"
              % (fit["per_iter_ns"] / 1e6, fit["launch_overhead_ns"] / 1e6,
                 fit["transfer_ns"] / 1e6), file=out)
        print("      fit r2 = %.4f over %d points, transfer flat to %.1f%%"
              % (fit["r2"], fit["points"], fit["transfer_spread"] * 100),
              file=out)
    if printed:
        print("", file=out)


def per_hash_fit(points):
    """
    Least squares of seconds-per-hash against iteration count.

    Every iteration is identical work by construction, so time per hash is
    linear in the iteration count on either side of the comparison:

        seconds_per_hash(N) = intercept + slope * N

    The slope is the cost of one iteration of compute. The intercept is
    everything paid once per hash regardless of N -- on a streaming device run
    that is the corpus upload and the launch, amortised over the batch, which
    is exactly the cost an accelerator has to earn back.

    Returns a dict, or None if the points cannot support a fit.
    """
    pts = [(n, 1.0 / hps) for n, hps in points if n > 0 and hps > 0]
    if len(pts) < 2:
        return None

    n_mean = sum(p[0] for p in pts) / len(pts)
    y_mean = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - n_mean) ** 2 for p in pts)
    if sxx == 0:
        return None                      # every point at the same N

    sxy = sum((p[0] - n_mean) * (p[1] - y_mean) for p in pts)
    slope = sxy / sxx
    intercept = y_mean - slope * n_mean

    ss_tot = sum((p[1] - y_mean) ** 2 for p in pts)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {"intercept": intercept, "slope": slope, "r2": r2,
            "points": len(pts)}


def break_even(device_points, cpu_points):
    """
    Solve for the iteration count at which offloading overtakes the whole CPU.

    This is the purchasing decision, and it is a different question from the
    PCIe balance point above. N* asks "is the bus in the way", which decides
    whether a *faster* accelerator would help. Break-even asks "does this
    accelerator beat the machine I already own", which decides whether to buy
    one at all. They can disagree in both directions: a device can be
    compute-limited and still slower end to end than the host, or bus-limited
    and still faster.

    The comparison is deliberately asymmetric. The device side is measured with
    `--transfer stream`, so its time includes getting the corpus across the
    link; the CPU side reads from the memory it already has. That asymmetry is
    the point -- it is what "should I offload this" actually costs.

    Both sides are linear in N, so the crossing is solved rather than searched:

        cpu:    p + q*N          device: c + d*N
        N_be = (c - p) / (q - d)

    q > d is the precondition -- the device must compute an iteration faster
    than the CPU does, or the curves diverge instead of crossing and no
    iteration count ever makes offloading pay.

    Returns a dict describing the outcome, or None if neither side can be fit.
    """
    dev = per_hash_fit(device_points)
    cpu = per_hash_fit(cpu_points)
    if not dev or not cpu:
        return None

    out = {
        "device_fit": dev,
        "cpu_fit": cpu,
        "device_per_iter_s": dev["slope"],
        "cpu_per_iter_s": cpu["slope"],
        "device_fixed_s": dev["intercept"],
        "cpu_fixed_s": cpu["intercept"],
        # What the ratio tends to as N grows and the transfer amortises away.
        "compute_advantage": (cpu["slope"] / dev["slope"]
                              if dev["slope"] > 0 else float("inf")),
    }

    # Time per hash at N=0 cannot be negative: it is the cost paid before any
    # iteration runs. A fit that extrapolates below zero has been dragged by a
    # point that is not on the line -- usually one degraded rung of the ladder,
    # thermal or contention -- and r2 does not catch it, because a single
    # outlier at the far end of a wide ladder can leave r2 above 0.98 while
    # tilting the slope enough to change the answer several-fold. Observed:
    # one bad 64-iteration CPU point turned a 1.2x compute advantage into 4.7x.
    out["warnings"] = []
    for side, fit in (("device", dev), ("CPU", cpu)):
        if fit["intercept"] < 0:
            out["warnings"].append(
                "%s fit extrapolates to a negative fixed cost (%.3f us) -- the "
                "ladder contains a point off the line, so the slope and the "
                "answer are both suspect" % (side, fit["intercept"] * 1e6))
        if fit["points"] < 3:
            out["warnings"].append(
                "%s fit has only %d points; r2 is 1.0 by construction"
                % (fit.get("label", "fit"), fit["points"]))
        elif fit["r2"] < 0.98:
            out["warnings"].append("%s fit is nonlinear (r2=%.4f)"
                                   % (side, fit["r2"]))

    denom = cpu["slope"] - dev["slope"]
    if denom <= 0:
        # The device computes no faster per iteration than the CPU, so more
        # iterations only widen the gap. Nothing to solve for.
        out["verdict"] = "never"
        out["n_break_even"] = None
        return out

    n_be = (dev["intercept"] - cpu["intercept"]) / denom
    out["n_break_even"] = n_be
    if n_be <= 1.0:
        # Already ahead at the smallest iteration count the workload allows.
        out["verdict"] = "always"
    else:
        out["verdict"] = "crosses"
    return out


def report_break_even(rows, out):
    """
    Print break-even per (algorithm, message size), where the grid has both a
    streaming device sweep and a CPU sweep over the same iteration ladder.

    At each iteration count both sides are taken at their best kernel, since
    the question is about the hardware rather than about a particular kernel.
    """
    groups = {}
    for r in rows:
        try:
            n = int(r["iterations"])
            hps = float(r["hashes_per_sec"])
        except (ValueError, KeyError, TypeError):
            continue
        key = (r["algorithm"], r["message_bytes"])
        side = "device" if r.get("runs_on") == "device" else "cpu"
        g = groups.setdefault(key, {"device": {}, "cpu": {}, "meta": {}})
        # Best kernel wins at each N: this is a hardware comparison.
        if hps > g[side].get(n, (0.0, ""))[0]:
            g[side][n] = (hps, r["kernel"])
        g["meta"].setdefault(side, r)

    printed = False
    for (alg, msg), g in sorted(groups.items()):
        if not g["device"] or not g["cpu"]:
            continue
        shared = sorted(set(g["device"]) & set(g["cpu"]))
        if len(shared) < 2:
            continue

        fit = break_even([(n, g["device"][n][0]) for n in shared],
                         [(n, g["cpu"][n][0]) for n in shared])
        if not fit:
            continue

        if not printed:
            print("\nCPU break-even  (device with transfers == whole CPU)",
                  file=out)
            printed = True

        dev_row, cpu_row = g["meta"]["device"], g["meta"]["cpu"]
        threads = cpu_row.get("threads", "?")
        label = "%s %s-byte" % (alg, msg)

        if fit["verdict"] == "never":
            print("  %-20s the device never overtakes the CPU -- it computes "
                  "an iteration %.2fx slower" % (label,
                  fit["device_per_iter_s"] / fit["cpu_per_iter_s"]), file=out)
        elif fit["verdict"] == "always":
            print("  %-20s the device is ahead at every iteration count "
                  "(break-even solves to %.2f, below the floor of 1)"
                  % (label, fit["n_break_even"]), file=out)
        else:
            print("  %-20s break-even at %.2f iterations"
                  % (label, fit["n_break_even"]), file=out)
            print("      offload pays from %d iterations up"
                  % max(1, math.ceil(fit["n_break_even"])), file=out)

        print("      device %s, CPU %s on %s threads"
              % (dev_row["kernel"], cpu_row["kernel"], threads), file=out)
        print("      per iteration: %.3f us device, %.3f us CPU  (%.2fx "
              "compute advantage once transfers amortise)"
              % (fit["device_per_iter_s"] * 1e6, fit["cpu_per_iter_s"] * 1e6,
                 fit["compute_advantage"]), file=out)
        print("      fixed cost per hash: %.3f us device (upload + launch), "
              "%.3f us CPU" % (fit["device_fixed_s"] * 1e6,
                               fit["cpu_fixed_s"] * 1e6), file=out)
        print("      fit r2 = %.4f device, %.4f CPU, over %d shared points"
              % (fit["device_fit"]["r2"], fit["cpu_fit"]["r2"], len(shared)),
              file=out)

        warn = list(fit["warnings"])
        if dev_row.get("transfer_mode") != "stream":
            warn.append("device rows are not streaming, so the link is not in "
                        "the timed region and this flatters the device")
        if str(threads) == "1":
            warn.append("CPU side is single-threaded, which understates the "
                        "machine you would be comparing against")
        if dev_row.get("working_set_bytes") != cpu_row.get("working_set_bytes"):
            warn.append("the two sides used different working sets")
        for w in warn:
            print("      [%s]" % w, file=out)

    if printed:
        print("", file=out)


def estimate_seconds(args, n_points):
    """Rough wall time per point, for the pre-flight estimate."""
    per = args.warmup_ms / 1000.0 + args.samples * args.time_ms / 1000.0
    if not args.kernel:
        per += 2.0            # autotune probes every available kernel
    per += 0.4                # corpus build + reference checksum, very rough
    return per * n_points


def human_table(rows, out):
    if not rows:
        return

    cols = [
        ("alg", "algorithm", 6),
        ("msg B", "message_bytes", 7),
        ("blk", "blocks_per_message", 5),
        ("iter", "iterations", 5),
        ("thr", "threads", 4),
        ("WS KiB", "working_set_kb", 9),
        ("kernel", "kernel", 11),
        ("MH/s", "hashes_per_sec", 11),
        ("MC/s", "compressions_per_sec", 10),
        ("MB/s", "message_bytes_per_sec", 10),
        ("CoV%", "cov_percent", 7),
        ("c/x", "compute_transfer_ratio", 7),
        ("bound", "bound_by", 9),
    ]

    # c/x and bound divide kernel time by transfer time, so they exist only for
    # a device kernel that moved data. A CPU-only sweep can never fill them,
    # and two permanently empty columns read as missing data rather than as
    # inapplicable ones. Drop any column no row has a value for.
    cols = [c for c in cols
            if any(str(r.get(c[1], "")).strip() for r in rows)]

    print("", file=out)
    print("  ".join(h.rjust(w) for h, _, w in cols), file=out)
    print("  ".join("-" * w for _, _, w in cols), file=out)

    for r in rows:
        cells = []
        for _, key, w in cols:
            v = r[key]
            if key in ("hashes_per_sec", "compressions_per_sec",
                       "message_bytes_per_sec"):
                v = "%.2f" % (float(v) / 1e6)
            cells.append(str(v).rjust(w))
        line = "  ".join(cells)
        if r["status"] != "ok":
            line += "   [%s]" % r["status"]
        print(line, file=out)
    print("", file=out)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep valubench across a parameter grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
axis syntax:
  55                 a single value
  55,1015,4087       an explicit list
  64:4096:*2         geometric: 64, 128, 256, ... up to 4096
  1:10:+3            arithmetic: 1, 4, 7, 10

examples:
  ./tools/sweep.py --message-bytes 55,1015,4087 > sweep.csv
  ./tools/sweep.py --kernel avx2-s4 --threads 1 \\
      --message-bytes 64:4096:*2 --iterations 1,4,16 --csv surface.csv
  ./tools/sweep.py --working-set-kb 64:262144:*4 --message-bytes 1015 --dry-run

CSV goes to stdout unless --csv names a file, in which case stdout gets the
table instead. Progress always goes to stderr, so redirecting stdout is safe.
""")

    ap.add_argument("--bin", default=None,
                    help="path to the valubench binary "
                         "(default: build/valubench beside this script)")
    ap.add_argument("--algorithm", default="md5", metavar="LIST",
                    help="md5, sha1, sha512 -- comma-separated (default md5)")
    ap.add_argument("--transfer", default="resident", metavar="LIST",
                    help="resident, stream -- comma-separated. 'stream' puts "
                         "the host-to-device upload inside the timed region; "
                         "sweep --iterations against it to find where compute "
                         "overtakes the link. No effect on CPU kernels.")
    ap.add_argument("--message-bytes", default="55", metavar="LIST")
    ap.add_argument("--iterations", default="1", metavar="LIST")
    ap.add_argument("--working-set-kb", default="1024", metavar="LIST")
    ap.add_argument("--threads", default=None, metavar="LIST",
                    help="default: one per online CPU")
    ap.add_argument("--kernel", default=None, metavar="LIST",
                    help="force kernels, comma-separated; skips autotune and "
                         "is much faster for large grids. 'md5/avx2-s4' pairs "
                         "only with its own algorithm; a bare 'avx2-s4' is "
                         "qualified per algorithm, so it sweeps one ISA across "
                         "all of them. Names are checked against the binary's "
                         "own registry.")
    ap.add_argument("--where", default="any", metavar="LIST",
                    help="restrict autotune to 'cpu' or 'device' kernels; "
                         "'any' (the default) lets them compete. A list makes "
                         "it an axis -- '--where device,cpu' over an iteration "
                         "ladder is what break-even needs, since it measures "
                         "both sides at identical parameters.")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--time-ms", type=int, default=100)
    ap.add_argument("--warmup-ms", type=int, default=300)
    ap.add_argument("--no-pin", dest="pin", action="store_false",
                    help="do not pin worker threads to cores")
    ap.add_argument("--no-precompute", action="store_true",
                    help="have every point compute its own reference checksum "
                         "instead of solving the iteration ladder once up "
                         "front. Slower by design; use it to check that the "
                         "precomputed answers and the per-point ones agree.")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-point timeout in seconds (default 900)")
    ap.add_argument("--resume", action="store_true",
                    help="skip points already present in the --csv file and "
                         "append to it, instead of starting over. For a grid "
                         "that died partway: a dropped connection, a preempted "
                         "instance, an hour of rented time running out.")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="write CSV here instead of stdout")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the grid and the time estimate, then exit")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a verification failure instead of "
                         "aborting")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress per-point progress")
    args = ap.parse_args()

    # Locate the binary relative to the repo, not the working directory.
    if args.bin is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.bin = os.path.join(os.path.dirname(here), "build", "valubench")
    if not (os.path.isfile(args.bin) and os.access(args.bin, os.X_OK)):
        found = shutil.which("valubench")
        if found:
            args.bin = found
        else:
            sys.exit("sweep: no valubench binary at %s -- run 'make' first, "
                     "or pass --bin" % args.bin)

    caps = Capabilities.query(args.bin)

    if args.resume and not args.csv:
        sys.exit("sweep: --resume needs --csv, since that is the file it resumes")

    if args.threads is None:
        args.threads = str(os.cpu_count() or 1)

    try:
        args.message_bytes = parse_list(args.message_bytes, "--message-bytes")
        args.iterations = parse_list(args.iterations, "--iterations")
        args.working_set_kb = parse_list(args.working_set_kb,
                                         "--working-set-kb")
        args.threads = parse_list(args.threads, "--threads")
        args.algorithm = parse_choice_list(args.algorithm, "--algorithm",
                                           caps.algorithms)
        args.transfer = parse_choice_list(args.transfer, "--transfer",
                                          caps.transfer_modes)
        args.where = parse_choice_list(args.where, "--where",
                                       caps.where_filters)
        args.kernel = (parse_choice_list(args.kernel, "--kernel", None)
                       if args.kernel else None)
    except ValueError as e:
        sys.exit("sweep: %s" % e)

    # The binary reports the ranges it enforces, so a grid that steps outside
    # them is caught here rather than as a usage error on every point.
    lim = caps.limits
    for value, low, high, flag in (
            (max(args.message_bytes), lim["message_bytes_min"],
             lim["message_bytes_max"], "--message-bytes"),
            (min(args.message_bytes), lim["message_bytes_min"],
             lim["message_bytes_max"], "--message-bytes"),
            (max(args.iterations), 1, lim["iterations_max"], "--iterations"),
            (max(args.threads), 0, lim["threads_max"], "--threads"),
            (args.samples, 1, lim["samples_max"], "--samples")):
        if value < low or value > high:
            sys.exit("sweep: %s %d is outside what this binary accepts "
                     "(%d..%d)" % (flag, value, low, high))

    points, skipped = build_grid(args, caps)

    for note in skipped:
        print("sweep: skipping %s" % note, file=sys.stderr)

    if not points:
        sys.exit("sweep: no valid points in the grid")

    est = estimate_seconds(args, len(points))
    print("sweep: %d point%s, roughly %s"
          % (len(points), "" if len(points) == 1 else "s",
             ("%.0f s" % est) if est < 120 else ("%.1f min" % (est / 60.0))),
          file=sys.stderr)
    if not args.kernel and len(points) > 8:
        print("sweep: pass --kernel to skip autotune on every point "
              "(much faster for large grids)", file=sys.stderr)

    if args.dry_run:
        for i, p in enumerate(points, 1):
            print("  %3d  %-7s %-9s kernel=%-14s message_bytes=%-7d "
                  "iterations=%-5d working_set_kb=%-8d threads=%d"
                  % (i, p["algorithm"], p["transfer"],
                     p["kernel"] or ("auto/" + p["where"]),
                     p["message_bytes"], p["iterations"],
                     p["working_set_kb"], p["threads"]), file=sys.stderr)
        return 0

    # --- resume ----------------------------------------------------------
    completed, append = set(), False
    if args.resume:
        if os.path.exists(args.csv) and os.path.getsize(args.csv) > 0:
            completed, header_ok = load_completed(args.csv)
            if not header_ok:
                sys.exit("sweep: %s has a different set of columns, so it came "
                         "from another version.\n       Move it aside rather "
                         "than mixing two schemas in one file." % args.csv)
            before = len(points)
            points = [p for p in points if point_id(p) not in completed]
            append = True
            print("sweep: resuming %s -- %d of %d points already done, %d to go"
                  % (args.csv, before - len(points), before, len(points)),
                  file=sys.stderr)
            if not points:
                print("sweep: nothing left to do", file=sys.stderr)
                return 0

    csv_target = (open(args.csv, "a" if append else "w", newline="")
                  if args.csv else sys.stdout)
    table_target = sys.stdout if args.csv else sys.stderr

    writer = csv.DictWriter(csv_target, fieldnames=CSV_COLUMNS,
                            extrasaction="ignore")
    if not append:
        writer.writeheader()

    refs = {} if args.no_precompute else precompute_references(args, points)
    if refs and not args.quiet:
        n = sum(len(v) for v in refs.values())
        print("  precomputed %d expected checksums in %d pass%s"
              % (n, len(refs), "" if len(refs) == 1 else "es"),
              file=sys.stderr)

    rows, failures, started = [], 0, time.time()

    try:
        for i, point in enumerate(points, 1):
            if not args.quiet:
                print("  [%d/%d] %s%s msg=%d iter=%d ws=%dK thr=%d ... "
                      % (i, len(points), point["algorithm"],
                         "/stream" if point["transfer"] == "stream" else "",
                         point["message_bytes"], point["iterations"],
                         point["working_set_kb"], point["threads"]),
                      end="", file=sys.stderr, flush=True)

            status, result = run_point(args, caps, point, refs)

            if result is None:
                failures += 1
                if not args.quiet:
                    print(status, file=sys.stderr)
                else:
                    print("sweep: point %d failed: %s" % (i, status),
                          file=sys.stderr)

                if status == "VERIFICATION FAILED" and not args.keep_going:
                    print("\nsweep: aborting. The hardware did not compute "
                          "correct MD5 digests, so every later point would be "
                          "suspect too. Pass --keep-going to override.",
                          file=sys.stderr)
                    return caps.exit["verify_failed"]
                continue

            row = row_from_result(result, status, point)
            rows.append(row)
            writer.writerow(row)
            csv_target.flush()

            if not args.quiet:
                print("%.2f MH/s  CoV %.2f%%%s"
                      % (float(row["hashes_per_sec"]) / 1e6,
                         float(row["cov_percent"]),
                         "" if status == "ok" else "  [%s]" % status),
                      file=sys.stderr)
    finally:
        if args.csv:
            csv_target.close()

    human_table(rows, table_target)
    report_balance(rows, table_target)
    report_break_even(rows, table_target)

    elapsed = time.time() - started
    print("sweep: %d/%d points in %.0f s%s"
          % (len(rows), len(points), elapsed,
             "" if failures == 0 else ", %d failed" % failures),
          file=sys.stderr)

    # The batch is rounded to a whole number of groups, and never goes below
    # VB_BATCH_LCM messages -- so several small --working-set-kb requests can
    # land on the same actual corpus. Say so rather than letting duplicate rows
    # look like independent measurements.
    seen, dupes = set(), 0
    for r in rows:
        key = (r["message_bytes"], r["iterations"], r["threads"],
               r["working_set_bytes"])
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        print("sweep: %d point%s collapsed onto an identical working set "
              "(the batch has a floor, so small --working-set-kb values round "
              "up to the same corpus); the working_set_kb column reports what "
              "was actually used" % (dupes, "" if dupes == 1 else "s"),
              file=sys.stderr)

    unstable = sum(1 for r in rows if r["stable"] == "false")
    if unstable:
        print("sweep: %d point%s exceeded the stability threshold; treat "
              "those as indicative only" % (unstable, "" if unstable == 1
                                            else "s"), file=sys.stderr)

    if args.csv:
        print("sweep: wrote %s" % args.csv, file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
