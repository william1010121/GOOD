#!/usr/bin/env python3
"""Exact cross-node merge for a finite epoch of GPU hash records.

Each input is a binary stream of little-endian (hi64, lo64, nonce) records.
The merge first builds the 56-bit prefix summary for each node, intersects the
two summaries, and only compares full records from shared prefixes.
"""
import argparse
import csv
import glob
import struct
from collections import defaultdict

RECORD = struct.Struct("<QQQ")


def read_records(paths):
    records = []
    for path in paths:
        with open(path, "rb") as fp:
            while True:
                data = fp.read(RECORD.size * 65536)
                if not data:
                    break
                if len(data) % RECORD.size:
                    raise ValueError(f"truncated record file: {path}")
                records.extend(RECORD.unpack_from(data, i)
                               for i in range(0, len(data), RECORD.size))
    return records


def common_prefix(a, b):
    x = a[0] ^ b[0]
    if x:
        return 64 - x.bit_length()
    y = a[1] ^ b[1]
    return 64 + (64 - y.bit_length()) if y else 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node0", required=True, nargs="+")
    ap.add_argument("--node1", required=True, nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    node0 = read_records(args.node0)
    node1 = read_records(args.node1)
    by0 = defaultdict(list)
    by1 = defaultdict(list)
    for r in node0:
        by0[r[0] >> 8].append(r)
    for r in node1:
        by1[r[0] >> 8].append(r)

    shared = sorted(by0.keys() & by1.keys())
    best = None
    for prefix in shared:
        for a in by0[prefix]:
            for b in by1[prefix]:
                bits = common_prefix(a, b)
                if best is None or bits > best[0]:
                    best = (bits, a[2], b[2], a, b)

    with open(args.output, "w", newline="") as fp:
        out = csv.writer(fp)
        out.writerow(("match_bits", "nonce_a", "nonce_b"))
        if best:
            out.writerow(best[:3])
    print(f"node0_records={len(node0)} node1_records={len(node1)}")
    print(f"shared_56bit_prefixes={len(shared)}")
    print(f"global_exact_best_qbit={best[0] if best else 0}")


if __name__ == "__main__":
    main()
