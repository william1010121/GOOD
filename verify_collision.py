#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# 隱藏題1 — SHA-256 部分碰撞 驗證工具
#
#   hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes 大端) )
#   計分 = 兩個雜湊開頭相同的 bit 數
#
#   python3 verify_collision.py -p <prefix> -a <nonce_a> -b <nonce_b>
#   python3 verify_collision.py --file solution_51.csv        # 直接讀繳交檔
#
# 相容性：不使用 f-string，Python 3.4 以上可執行。
# =====================================================================
from __future__ import print_function
import argparse
import csv
import hashlib
import struct

MAX_NONCE = 2 ** 64 - 1


def sha_hex(prefix, nonce):
    # >Q = 8 bytes 大端無號整數
    return hashlib.sha256(prefix.encode("ascii") + struct.pack(">Q", nonce)).hexdigest()


def common_prefix_bits(hex_a, hex_b):
    x = int(hex_a, 16) ^ int(hex_b, 16)
    return 256 - x.bit_length()          # x == 0 時為 256（完全碰撞）


def check(prefix, a, b):
    if a == b:
        print("!! nonce_a 與 nonce_b 相同，不合法（必須是兩個不同的 nonce）")
        return None
    for name, v in (("nonce_a", a), ("nonce_b", b)):
        if v < 0 or v > MAX_NONCE:
            print("!! %s 超出 64-bit 範圍" % name)
            return None

    ha, hb = sha_hex(prefix, a), sha_hex(prefix, b)
    bits = common_prefix_bits(ha, hb)
    same_hex = bits // 4                 # 完整相同的 16 進位字元數

    print("prefix   : %s" % prefix)
    print("nonce_a  : %d" % a)
    print("nonce_b  : %d" % b)
    print("hash_a   : %s" % ha)
    print("hash_b   : %s" % hb)
    print("           %s" % ("^" * same_hex))
    print("共同前綴 : %d bits（前 %d 個 16 進位字元相同）" % (bits, same_hex))
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prefix", help="固定字串 prefix")
    ap.add_argument("-a", "--nonce_a", type=int)
    ap.add_argument("-b", "--nonce_b", type=int)
    ap.add_argument("--file", help="直接讀取繳交的 CSV 檔")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as f:
            row = list(csv.DictReader(f))[0]
        bits = check(row["prefix"], int(row["nonce_a"]), int(row["nonce_b"]))
        claimed = row.get("match_bits")
        if bits is not None and claimed is not None:
            ok = (bits == int(claimed))
            print("聲稱 bits : %s   %s" % (claimed, "一致" if ok else "!! 不一致"))
        return

    if args.prefix is None or args.nonce_a is None or args.nonce_b is None:
        ap.error("請提供 -p/-a/-b，或用 --file 讀繳交檔")
    check(args.prefix, args.nonce_a, args.nonce_b)


if __name__ == "__main__":
    main()
